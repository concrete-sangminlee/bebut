"""SFT(지도 미세조정) — 베이스 모델을 대화형 모델로.

핵심:
- chat template: <|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>
- loss masking: assistant 응답 토큰에만 loss (나머지 target = -100 → cross_entropy가 무시)
- 데이터: 공개 한국어 instruction 데이터셋 (KoAlpaca v1.1a, KULLM v2)

사용 예:
  python sft.py --base checkpoints/base_1b/latest.pt --config sft_1b
"""

import argparse
import importlib
import os
import random
import time

import torch

from model import GPT
from tokenizer.bpe import ByteBPETokenizer

DATASETS = [
    # (HF dataset, instruction 필드, input 필드, output 필드)
    ("beomi/KoAlpaca-v1.1a", "instruction", None, "output"),
    ("nlpai-lab/kullm-v2", "instruction", "input", "output"),
]


def build_examples(tok, max_len, limit=None):
    """대화를 토큰 id + loss mask로 변환한 예제 리스트를 만든다."""
    from datasets import load_dataset

    im_start = tok.special_tokens["<|im_start|>"]
    im_end = tok.special_tokens["<|im_end|>"]
    examples = []
    for name, f_inst, f_input, f_out in DATASETS:
        ds = load_dataset(name, split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        print(f"[{name}] {len(ds)}개 예제")
        for row in ds:
            user = row[f_inst]
            if f_input and row.get(f_input):
                user += "\n\n" + row[f_input]
            assistant = row[f_out]
            if not user or not assistant:
                continue
            # user 파트 (loss 없음)
            # 형식은 sample.py --chat, annealing instruction 데이터와 정확히 같아야 한다:
            #   <|im_start|>user\n{질문}<|im_end|>\n<|im_start|>assistant\n{답}<|im_end|>
            ctx = ([im_start] + tok.encode(f"user\n{user}") + [im_end] + tok.encode("\n")
                   + [im_start] + tok.encode("assistant\n"))
            # assistant 파트 (loss 있음)
            ans = tok.encode(assistant) + [im_end]
            ids = (ctx + ans)[:max_len]
            if len(ids) < len(ctx) + 4:  # 응답이 거의 다 잘리면 버림
                continue
            mask = [0] * len(ctx) + [1] * (len(ids) - len(ctx))
            examples.append((ids, mask))
    random.Random(42).shuffle(examples)
    return examples


def collate(batch, pad_id):
    """가변 길이 예제를 오른쪽 패딩. x는 pad, y는 -100(무시)으로 채운다."""
    L = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), L - 1), pad_id, dtype=torch.long)
    y = torch.full((len(batch), L - 1), -100, dtype=torch.long)
    for i, (ids, mask) in enumerate(batch):
        t = torch.tensor(ids)
        x[i, : len(ids) - 1] = t[:-1]
        # next-token 예측: 위치 j의 target은 ids[j+1], assistant 구간만 살림
        for j in range(len(ids) - 1):
            if mask[j + 1]:
                y[i, j] = ids[j + 1]
    return x, y


@torch.no_grad()
def val_loss(model, val_ex, pad_id, B, device, device_type, loss_chunk):
    """assistant 응답 토큰에 대한 평균 loss (학습에 안 쓴 예제)."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val_ex) - B + 1, B):
        x, y = collate(val_ex[i : i + B], pad_id)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y, loss_chunk=loss_chunk)
        tot += loss.item()
        n += 1
    model.train()
    return tot / max(1, n)


def save(model, step, config, ckpt_dir):
    """원자적 저장 — 임시 파일에 쓴 뒤 rename."""
    path = os.path.join(ckpt_dir, "latest.pt")
    torch.save({"model": model.state_dict(), "step": step, "config": config}, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="사전학습 체크포인트")
    ap.add_argument("--config", default="sft_1b")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--limit", type=int, default=None, help="데이터셋별 예제 수 제한 (테스트용)")
    args = ap.parse_args()

    conf = importlib.import_module(f"configs.{args.config}")
    tcfg = conf.train
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu" if device == "cpu" else "mps"

    tok = ByteBPETokenizer.load(args.tokenizer)
    examples = build_examples(tok, tcfg["max_len"], args.limit)
    n_val = max(64, len(examples) // 100)
    val_ex, train_ex = examples[:n_val], examples[n_val:]
    print(f"train {len(train_ex)} / val {len(val_ex)} 예제")

    # CPU로 읽는다. 사전학습 체크포인트엔 옵티마이저 상태(1.2B 기준 약 10GB)도 들어 있어서
    # GPU로 바로 올리면 그만큼이 학습 내내 GPU에 남는다.
    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    model = GPT(conf.model)
    model.load_state_dict(ckpt["model"])
    print(f"베이스 모델: {args.base} (step {ckpt['step']})")
    del ckpt
    model = model.to(device)
    opt = torch.optim.AdamW(model.param_groups(tcfg["weight_decay"]),
                            lr=tcfg["max_lr"], betas=(0.9, 0.95), fused=(device == "cuda"))

    ckpt_dir = os.path.join("checkpoints", args.config)
    os.makedirs(ckpt_dir, exist_ok=True)
    pad_id = tok.special_tokens["<|pad|>"]
    B = tcfg["batch_size"]
    steps_per_epoch = len(train_ex) // (B * tcfg["grad_accum"])
    max_steps = steps_per_epoch * tcfg["epochs"]
    print(f"총 {max_steps} steps ({tcfg['epochs']} epochs)")

    def batches(data):
        for i in range(0, len(data) - B + 1, B):
            yield collate(data[i : i + B], pad_id)

    vl0 = val_loss(model, val_ex, pad_id, B, device, device_type, tcfg.get("loss_chunk"))
    print(f"SFT 전 val loss (베이스 모델): {vl0:.4f}", flush=True)

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(tcfg["epochs"]):
        it = batches(train_ex)
        for step_in_epoch in range(steps_per_epoch):
            # linear warmup → cosine decay
            import math

            if step < tcfg["warmup_steps"]:
                lr = tcfg["max_lr"] * (step + 1) / tcfg["warmup_steps"]
            else:
                frac = (step - tcfg["warmup_steps"]) / max(1, max_steps - tcfg["warmup_steps"])
                lr = tcfg["min_lr"] + 0.5 * (tcfg["max_lr"] - tcfg["min_lr"]) * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr

            loss_acc = 0.0
            for _ in range(tcfg["grad_accum"]):
                x, y = next(it)
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y, loss_chunk=tcfg.get("loss_chunk"))
                (loss / tcfg["grad_accum"]).backward()
                loss_acc += loss.item() / tcfg["grad_accum"]
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % 10 == 0:
                print(f"epoch {epoch} step {step}/{max_steps} | loss {loss_acc:.4f} | lr {lr:.2e} "
                      f"| {time.time()-t0:.0f}s", flush=True)
            if step > 0 and step % tcfg["save_every"] == 0 or step == max_steps - 1:
                print(f"체크포인트 저장: {save(model, step, args.config, ckpt_dir)} (step {step})",
                      flush=True)
            step += 1

        vl = val_loss(model, val_ex, pad_id, B, device, device_type, tcfg.get("loss_chunk"))
        print(f"=== epoch {epoch} 종료 | val loss {vl:.4f} ===", flush=True)
        save(model, step - 1, args.config, ckpt_dir)

    print("SFT 완료 — python sample.py --ckpt", os.path.join(ckpt_dir, "latest.pt"), "--chat")


if __name__ == "__main__":
    main()
