"""DPO(Direct Preference Optimization) — SFT 모델을 '더 나은 답' 쪽으로 미는 정렬 단계.

SFT는 정답 예시를 따라 하게 할 뿐 '어떤 답이 더 나은가'를 가르치지 않는다. 같은 질문에
대한 좋은 답(chosen)과 나쁜 답(rejected) 쌍이 있으면, DPO는 보상 모델이나 강화학습 없이
분류 문제 하나로 그 선호를 학습한다 (Rafailov et al., 2023):

    L = -log σ( β · [ (log π(y_c|x) - log π_ref(y_c|x)) - (log π(y_r|x) - log π_ref(y_r|x)) ] )

π는 학습 중인 모델, π_ref는 시작점(SFT 모델)을 얼린 것이다. 참조 모델과의 차이로
재기 때문에, 모델이 원래 잘 쓰던 표현을 망가뜨리지 않으면서 상대적인 선호만 옮긴다.
β가 클수록 참조 모델에서 덜 벗어난다.

참조 모델은 메모리에 두지 않는다. π_ref는 학습 내내 바뀌지 않으므로, 학습 전에 정책
모델(= 시작점) 자신으로 모든 쌍의 log π_ref를 한 번 계산해 저장한다. 참조 모델을 따로
bf16으로 들고 있으면(GPU 2.4GB) 정책(fp32 가중치 + autocast)과 연산 정밀도가 달라
시작부터 쌍마다 0.04~0.5 nats씩 어긋나고, 그게 모든 쌍에 고정된 가짜 보상으로 들어간다.

- log π(y|x)는 응답 토큰의 로그확률 합. 문맥(질문) 토큰은 세지 않는다.
- chosen과 rejected를 한 배치로 묶어 forward 한 번.
- 64K vocab이라 응답 위치의 hidden만 골라 lm_head를 청크 단위로 (gradient checkpointing).
- 시작 시 정책 = 참조이므로 loss는 정확히 ln 2 ≈ 0.693 — 학습이 제대로 도는지 첫 확인점.

데이터 (JSONL 한 줄, 셋 중 하나):
  {"prompt": "질문", "chosen": "좋은 답", "rejected": "나쁜 답"}
  {"messages": [...이전 대화...], "chosen": "좋은 답", "rejected": "나쁜 답"}
  {"chosen": [...대화..., {"role": "assistant", ...}], "rejected": [...같은 대화..., {...}]}
      (UltraFeedback-binarized 형식: 마지막 assistant 턴만 다르다)
  --data hf:<dataset> 이면 HuggingFace 데이터셋의 train split을 같은 규칙으로 읽는다.

사용:
  python dpo.py --base checkpoints/sft_1b/latest.pt --config dpo_1b --data prefs.jsonl
"""

import argparse
import importlib
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from chat import encode_prompt
from model import GPT
from sample import load_checkpoint, pick_device
from tokenizer.bpe import ByteBPETokenizer

# ---------------- 데이터 ----------------


def parse_pair(row):
    """한 행 → (문맥 messages, chosen 답, rejected 답). 형식이 안 맞으면 None."""
    c, r = row.get("chosen"), row.get("rejected")
    if isinstance(c, list) and isinstance(r, list):
        if not c or not r or c[-1].get("role") != "assistant" or r[-1].get("role") != "assistant":
            return None
        if c[:-1] != r[:-1]:  # 문맥이 다르면 같은 질문에 대한 비교가 아니다
            return None
        ctx, c, r = c[:-1], c[-1].get("content"), r[-1].get("content")
    elif "messages" in row:
        ctx = row["messages"]
    elif row.get("prompt"):
        ctx = [{"role": "user", "content": row["prompt"]}]
    else:
        return None
    if not ctx or not isinstance(c, str) or not isinstance(r, str) or not c or not r or c == r:
        return None
    return [{"role": m["role"], "content": m["content"]} for m in ctx], c, r


def encode_pair(tok, ctx, chosen, rejected, max_len, min_resp=4):
    """(문맥 ids, chosen 응답 ids, rejected 응답 ids). 응답은 <|im_end|>까지 포함.

    문맥 + 응답 = chat.encode_messages(문맥 + [assistant 응답])와 토큰 단위로 같다
    (SFT 때와 같은 형식). 너무 길면 문맥은 앞에서, 응답은 뒤에서 자른다.
    """
    end = tok.special_tokens["<|im_end|>"]
    p = encode_prompt(tok, ctx)
    cr = [tok.encode(x) + [end] for x in (chosen, rejected)]
    room = max_len - max(len(x) for x in cr)
    if room < len(p):  # 응답이 길면 문맥을 앞에서 잘라 응답 자리를 확보 (최대 절반)
        p = p[-max(max_len // 2, room):]
    cr = [x[: max_len - len(p)] for x in cr]
    if min(len(x) for x in cr) < min_resp:
        return None
    return p, cr[0], cr[1]


def iter_rows(path, limit=None):
    if path.startswith("hf:"):
        from datasets import load_dataset

        ds = load_dataset(path[3:], split="train")
        rows = ds if limit is None else ds.select(range(min(limit, len(ds))))
        yield from rows
        return
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            yield json.loads(line)


def build_pairs(tok, path, max_len, limit=None):
    pairs, skipped = [], 0
    for row in iter_rows(path, limit):
        parsed = parse_pair(row)
        enc = encode_pair(tok, *parsed, max_len) if parsed else None
        if enc:
            pairs.append(enc)
        else:
            skipped += 1
    random.Random(42).shuffle(pairs)
    return pairs, skipped


def collate_pairs(batch, pad_id):
    """B개 쌍 → 2B개 시퀀스 (앞 B개 chosen, 뒤 B개 rejected). y는 응답 위치만 target."""
    seqs = [(p, c) for p, c, _ in batch] + [(p, r) for p, _, r in batch]
    L = max(len(p) + len(x) for p, x in seqs) - 1
    x = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    y = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, (p, resp) in enumerate(seqs):
        ids = torch.tensor(p + resp)
        x[i, : len(ids) - 1] = ids[:-1]
        y[i, len(p) - 1 : len(ids) - 1] = ids[len(p):]  # 위치 t의 출력이 토큰 t+1을 예측
    return x, y


# ---------------- 로그확률 · loss ----------------


def _token_logprobs(h, t, w):
    return -F.cross_entropy(F.linear(h, w).float(), t, reduction="none")


def sequence_logprobs(model, x, y, chunk=4096):
    """시퀀스별 응답 토큰 로그확률 합 (2B,). y == -100인 위치는 세지 않는다.

    응답 위치의 hidden만 골라 lm_head를 청크로 적용하고, 학습 중이면 청크마다
    gradient checkpointing으로 logits를 backward 때 다시 만든다 (64K vocab logits는
    토큰 수천 개만 돼도 수 GB다).
    """
    h = model.hidden_states(x)
    rows, cols = (y != -100).nonzero(as_tuple=True)
    hs, ts = h[rows, cols], y[rows, cols]
    w = model.lm_head.weight
    parts = []
    for i in range(0, hs.size(0), chunk):
        hc, tc = hs[i : i + chunk], ts[i : i + chunk]
        if torch.is_grad_enabled() and hc.requires_grad:
            parts.append(torch.utils.checkpoint.checkpoint(_token_logprobs, hc, tc, w,
                                                           use_reentrant=False))
        else:
            parts.append(_token_logprobs(hc, tc, w))
    lp = torch.cat(parts) if parts else hs.new_zeros(0, dtype=torch.float32)
    return torch.zeros(x.size(0), dtype=torch.float32, device=x.device).index_add(0, rows, lp)


def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta, label_smoothing=0.0):
    """시그모이드 DPO loss와 진단값. label_smoothing>0이면 선호 라벨 노이즈를 가정(cDPO)."""
    logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    loss = -(F.logsigmoid(logits) * (1 - label_smoothing) + F.logsigmoid(-logits) * label_smoothing)
    rc, rr = beta * (pol_c - ref_c).detach(), beta * (pol_r - ref_r).detach()
    return loss.mean(), {"reward_chosen": rc.mean().item(), "reward_rejected": rr.mean().item(),
                         "margin": (rc - rr).mean().item(), "accuracy": (rc > rr).float().mean().item()}


def policy_logprobs(policy, batch, pad_id, device, device_type):
    """(chosen 로그확률, rejected 로그확률), 각 (B,)."""
    x, y = collate_pairs(batch, pad_id)
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        p = sequence_logprobs(policy, x, y)
    return p[: len(batch)], p[len(batch):]


@torch.no_grad()
def reference_logprobs(model, pairs, pad_id, B, device, device_type) -> torch.Tensor:
    """학습 시작 전 모델로 모든 쌍의 log π_ref를 계산 → (N, 2) [chosen, rejected].

    학습 때와 같은 함수·같은 autocast로 계산하므로 step 0의 정책과 정확히 같은 값이다.
    """
    model.eval()
    out = []
    for i in range(0, len(pairs), B):
        c, r = policy_logprobs(model, pairs[i : i + B], pad_id, device, device_type)
        out.append(torch.stack([c, r], dim=1).float().cpu())
    model.train()
    return torch.cat(out) if out else torch.zeros(0, 2)


def pair_loss(policy, batch, refs, pad_id, device, device_type, beta, ls):
    """refs: (B, 2) 미리 계산한 참조 로그확률."""
    c, r = policy_logprobs(policy, batch, pad_id, device, device_type)
    refs = refs.to(c.device)
    return dpo_loss(c, r, refs[:, 0], refs[:, 1], beta, ls)


@torch.no_grad()
def evaluate(policy, pairs, refs, pad_id, B, device, device_type, beta, ls):
    policy.eval()
    tot = {"loss": 0.0, "reward_chosen": 0.0, "reward_rejected": 0.0, "margin": 0.0, "accuracy": 0.0}
    n = 0
    for i in range(0, len(pairs), B):
        batch = pairs[i : i + B]
        loss, st = pair_loss(policy, batch, refs[i : i + B], pad_id, device, device_type, beta, ls)
        k = len(batch)
        tot["loss"] += loss.item() * k
        for key, v in st.items():
            tot[key] += v * k
        n += k
    policy.train()
    return {k: v / max(1, n) for k, v in tot.items()}


def save(model, step, config, path):
    torch.save({"model": model.state_dict(), "step": step, "config": config}, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="시작점이자 참조 모델 (보통 SFT 체크포인트)")
    ap.add_argument("--config", default="dpo_1b")
    ap.add_argument("--data", required=True, help="선호 쌍 JSONL 또는 hf:<dataset>")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    conf = importlib.import_module(f"configs.{args.config}")
    tcfg = conf.train
    device = pick_device(args.device)
    device_type = "cuda" if device == "cuda" else "cpu" if device == "cpu" else "mps"
    torch.manual_seed(tcfg.get("seed", 1337))

    tok = ByteBPETokenizer.load(args.tokenizer)
    pairs, skipped = build_pairs(tok, args.data, tcfg["max_len"], args.limit)
    n_val = min(max(32, len(pairs) // 50), len(pairs) // 5)
    val, train = pairs[:n_val], pairs[n_val:]
    print(f"선호 쌍 train {len(train)} / val {len(val)} (형식 불일치·너무 짧아 버린 행 {skipped})")

    ckpt = load_checkpoint(args.base)
    policy = GPT(conf.model)
    policy.load_state_dict(ckpt["model"])
    print(f"시작 모델: {args.base} (step {ckpt.get('step')})")
    del ckpt
    policy = policy.to(device)
    opt = torch.optim.AdamW(policy.param_groups(tcfg["weight_decay"]), lr=tcfg["max_lr"],
                            betas=(0.9, 0.95), fused=(device == "cuda"))

    pad_id = tok.special_tokens["<|pad|>"]
    B, accum = tcfg["batch_size"], tcfg["grad_accum"]
    beta, ls = tcfg["beta"], tcfg.get("label_smoothing", 0.0)
    steps_per_epoch = len(train) // (B * accum)
    max_steps = steps_per_epoch * tcfg["epochs"]
    assert max_steps > 0, "선호 쌍이 step 하나 분량도 안 된다"
    warmup = max(1, int(max_steps * tcfg.get("warmup_frac", 0.1)))
    ckpt_dir = os.path.join(args.out_dir, args.config)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"총 {max_steps} steps | β={beta} | step당 {B * accum}쌍")

    t_ref = time.time()
    train_ref = reference_logprobs(policy, train, pad_id, B, device, device_type)
    val_ref = reference_logprobs(policy, val, pad_id, B, device, device_type)
    print(f"참조 로그확률 계산 {len(train) + len(val)}쌍, {time.time() - t_ref:.0f}초", flush=True)
    v0 = evaluate(policy, val, val_ref, pad_id, B, device, device_type, beta, ls)
    print(f"시작 val: loss {v0['loss']:.4f} (정책=참조라 ln2={math.log(2):.4f}여야 한다)", flush=True)

    step, t0 = 0, time.time()
    policy.train()
    for epoch in range(tcfg["epochs"]):
        order = list(range(len(train)))
        random.Random(1000 + epoch).shuffle(order)
        for s in range(steps_per_epoch):
            # warmup 후 cosine decay — DPO는 학습률에 민감해 SFT보다 한참 낮게 쓴다
            if step < warmup:
                lr = tcfg["max_lr"] * (step + 1) / warmup
            else:
                frac = (step - warmup) / max(1, max_steps - warmup)
                lr = tcfg["max_lr"] * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr
            agg = {"loss": 0.0, "accuracy": 0.0, "margin": 0.0}
            for a in range(accum):
                idx = order[(s * accum + a) * B : (s * accum + a + 1) * B]
                loss, st = pair_loss(policy, [train[i] for i in idx], train_ref[idx], pad_id,
                                     device, device_type, beta, ls)
                (loss / accum).backward()  # 쌍 수가 같으니 단순 평균이 곧 쌍 평균
                agg["loss"] += loss.item() / accum
                agg["accuracy"] += st["accuracy"] / accum
                agg["margin"] += st["margin"] / accum
            torch.nn.utils.clip_grad_norm_(policy.parameters(), tcfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % tcfg.get("log_every", 10) == 0:
                print(f"epoch {epoch} step {step}/{max_steps} | loss {agg['loss']:.4f} "
                      f"| 보상 정확도 {agg['accuracy']:.2f} | 마진 {agg['margin']:.3f} | lr {lr:.2e} "
                      f"| {time.time() - t0:.0f}s", flush=True)
            if step > 0 and step % tcfg["save_every"] == 0:
                save(policy, step, args.config, os.path.join(ckpt_dir, "latest.pt"))
            step += 1
        v = evaluate(policy, val, val_ref, pad_id, B, device, device_type, beta, ls)
        print(f"=== epoch {epoch} 종료 | val loss {v['loss']:.4f} | 보상 정확도 {v['accuracy']:.3f} "
              f"| 마진 {v['margin']:.3f} (chosen {v['reward_chosen']:+.3f} / "
              f"rejected {v['reward_rejected']:+.3f}) ===", flush=True)
        save(policy, step - 1, args.config, os.path.join(ckpt_dir, f"epoch{epoch}.pt"))
        save(policy, step - 1, args.config, os.path.join(ckpt_dir, "latest.pt"))
    print("DPO 완료 — python sample.py --ckpt", os.path.join(ckpt_dir, "latest.pt"), "--chat")


if __name__ == "__main__":
    main()
