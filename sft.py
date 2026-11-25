"""SFT(지도 미세조정) — 베이스 모델을 대화형 모델로.

핵심:
- chat template: chat.py 하나에서 정의 (학습·추론·평가가 공유)
- loss masking: assistant 응답 토큰에만 loss (나머지 target = -100 → cross_entropy가 무시)
- 데이터: 공개 한국어 instruction 데이터셋 (KoAlpaca v1.1a, KULLM v2) 또는 --data JSONL
- gradient accumulation은 **토큰 가중** — 마이크로배치 평균을 그냥 평균내면 짧은 답이
  많은 배치의 토큰이 긴 답 배치의 토큰보다 몇 배 큰 가중치를 받는다
- 길이별 묶음(group_by_length): 비슷한 길이끼리 배치를 만들어 패딩 낭비를 줄인다
- 에폭마다 epoch{n}.pt를 남긴다 — 1.2B SFT에선 에폭0이 val loss·반복률 모두 더 나았다
- 재개: save_every마다, 에폭 끝마다 resume.pt(가중치 + 옵티마이저 + 위치)를 남긴다. 죽으면
  --resume으로 그 자리에서 잇는다 — 중단 없이 학습한 것과 가중치가 비트 단위로 같다. 예전엔
  1.2B SFT(5.5시간)가 5시간째에 죽으면 처음부터 다시 해야 했다 (latest.pt를 --base로 주면
  학습률 스케줄과 AdamW 모멘트가 초기화돼 다른 모델이 된다)

사용 예:
  python sft.py --base checkpoints/base_1b/latest.pt --config sft_1b
  python sft.py --base ... --config sft_1b --data my_chats.jsonl   # 직접 만든 데이터
  python sft.py --base ... --config sft_1b --extra-data data/math_sft.jsonl   # 기본 데이터 + 산술
  python sft.py --config sft_1b --resume checkpoints/sft_1b/resume.pt   # 중단된 학습 잇기 (데이터 인자는 똑같이)

--data JSONL 한 줄 형식 (둘 중 하나):
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  {"instruction": "...", "input": "...(선택)", "output": "..."}
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import time
from array import array

import torch

from chat import encode_messages, instruction_to_messages
from model import GPT
from sample import device_type_of, load_checkpoint, pick_device
from tokenizer.bpe import ByteBPETokenizer
from train import step_if_finite

DATASETS = [
    # (HF dataset, instruction 필드, input 필드, output 필드)
    ("beomi/KoAlpaca-v1.1a", "instruction", None, "output"),
    ("nlpai-lab/kullm-v2", "instruction", "input", "output"),
]


def example_from_messages(tok, messages, max_len):
    """messages → (ids, mask). 응답이 거의 다 잘리면 None."""
    ids, mask = encode_messages(tok, messages, add_generation_prompt=False)
    ids, mask = ids[:max_len], mask[:max_len]
    if sum(mask) < 4:  # 응답 토큰이 4개도 안 남으면 버림 (예전 규칙과 같다)
        return None
    return ids, mask


def _instruction_row(instruction, inp, output):
    """예전 build_examples와 같은 채택 규칙: 결합한 user 문자열과 답이 비어 있지 않으면 쓴다.

    instruction이 비어도 input이 있으면 user = "\n\n" + input으로 남는다. 이 규칙이
    바뀌면 예제 수가 달라져 Random(42) 셔플 결과, 곧 train/val 분할 전체가 달라진다.
    """
    msgs = instruction_to_messages(instruction or "", output or "", inp)
    return msgs if msgs[0]["content"] and output else None


def iter_conversations(data_path=None, limit=None):
    """messages 목록을 차례로 내놓는다. data_path가 없으면 기본 HF 데이터셋."""
    if data_path:
        with open(data_path, encoding="utf-8") as f:
            for n, line in enumerate(f):
                if limit and n >= limit:
                    break
                row = json.loads(line)
                if "messages" in row:
                    yield row["messages"]
                else:
                    msgs = _instruction_row(row.get("instruction"), row.get("input"), row.get("output"))
                    if msgs:
                        yield msgs
        return

    from datasets import load_dataset

    for name, f_inst, f_input, f_out in DATASETS:
        ds = load_dataset(name, split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        print(f"[{name}] {len(ds)}개 예제")
        for row in ds:
            msgs = _instruction_row(row[f_inst], row.get(f_input) if f_input else None, row[f_out])
            if msgs:
                yield msgs


def build_examples(tok, max_len, limit=None, data_path=None):
    """대화를 토큰 id + loss mask로 변환한 예제 리스트를 만든다."""
    examples = []
    for msgs in iter_conversations(data_path, limit):
        ex = example_from_messages(tok, msgs, max_len)
        if ex:
            examples.append(ex)
    random.Random(42).shuffle(examples)
    return examples


def collate(batch, pad_id):
    """가변 길이 예제를 오른쪽 패딩. x는 pad, y는 -100(무시)으로 채운다."""
    L = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), L - 1), pad_id, dtype=torch.long)
    y = torch.full((len(batch), L - 1), -100, dtype=torch.long)
    for i, (ids, mask) in enumerate(batch):
        t = torch.tensor(ids)
        m = torch.tensor(mask[1:], dtype=torch.bool)
        x[i, : len(ids) - 1] = t[:-1]
        # next-token 예측: 위치 j의 target은 ids[j+1], assistant 구간만 살림
        y[i, : len(ids) - 1] = torch.where(m, t[1:], torch.tensor(-100))
    return x, y


def make_batches(data, B, seed, group_by_length, group_size=64):
    """에폭별 배치 순서. group_by_length면 group_size개 배치 분량씩 길이로 정렬해 묶는다.

    전체를 길이순으로 정렬하면 배치마다 길이가 달라 학습이 길이 순서로 진행되는
    편향이 생긴다. 무작위로 섞은 뒤 작은 덩어리 안에서만 정렬하고, 배치 순서를 다시
    섞으면 패딩은 크게 줄면서 순서는 무작위에 가깝게 남는다.
    """
    rng = random.Random(seed)
    order = list(range(len(data)))
    rng.shuffle(order)
    batches = []
    chunk = B * group_size if group_by_length else B
    for i in range(0, len(order), chunk):
        part = order[i : i + chunk]
        if group_by_length:
            part.sort(key=lambda j: len(data[j][0]))
        batches += [part[k : k + B] for k in range(0, len(part) - B + 1, B)]
    rng.shuffle(batches)
    return batches


def padding_efficiency(data, batches) -> float:
    real = sum(len(data[j][0]) for b in batches for j in b)
    padded = sum(max(len(data[j][0]) for j in b) * len(b) for b in batches)
    return real / max(1, padded)


@torch.no_grad()
def val_loss(model, val_ex, pad_id, B, device, device_type, loss_chunk):
    """assistant 응답 토큰에 대한 평균 loss (학습에 안 쓴 예제, 토큰 가중)."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val_ex) - B + 1, B):
        x, y = collate(val_ex[i : i + B], pad_id)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y, loss_chunk=loss_chunk)
        k = int((y != -100).sum())
        tot += loss.item() * k
        n += k
    model.train()
    return tot / max(1, n)


def save(model, step, config, path):
    """원자적 저장 — 임시 파일에 쓴 뒤 rename."""
    torch.save({"model": model.state_dict(), "step": step, "config": config}, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


RESUME_FILE = "resume.pt"
TRAJECTORY_FREE = ("save_every", "log_every")  # 바꿔도 학습 결과가 달라지지 않는 설정


def data_fingerprint(examples, tcfg) -> str:
    """학습 데이터(토큰 id 전부)와 학습 설정의 지문. 재개할 때 다르면 거절한다 — 데이터나 배치
    구성이 바뀐 채 이어가면 위치(에폭 안 몇 번째 step)가 다른 예제를 가리킨다."""
    h = hashlib.sha256(json.dumps({k: v for k, v in tcfg.items() if k not in TRAJECTORY_FREE},
                                  sort_keys=True).encode())
    h.update(str(len(examples)).encode())
    for ex in examples:  # SFT: (ids, mask) · DPO: (문맥, chosen, rejected)
        for part in ex:
            h.update(array("i", part).tobytes())
    return h.hexdigest()[:16]


def save_resume(ckpt_dir, model, opt, **state):
    """재개용 전체 상태를 원자적으로 저장. state: config · fingerprint · epoch · step_in_epoch(다음에
    돌 자리) · step(다음 전역 step) · bad(NaN 가드 연속 횟수) + 스크립트별 추가 값."""
    path = os.path.join(ckpt_dir, RESUME_FILE)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), **state}, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def refuse_stale_resume(ckpt_dir):
    """새로 시작(--base)하는데 출력 폴더에 이전 실행의 resume.pt가 남아 있으면 멈춘다.

    그대로 두면, 새 실행이 첫 저장(save_every) 전에 죽었을 때 --resume이 '이전' 실행을 조용히
    이어 간다 — 지문은 데이터와 설정만 보므로 시작 가중치가 달라도 통과한다."""
    path = os.path.join(ckpt_dir, RESUME_FILE)
    if os.path.exists(path):
        raise SystemExit(f"{path}가 남아 있습니다 (중단된 이전 실행의 재개 상태). 이어서 하려면 --base 대신 "
                         f"--resume {path}, 새로 시작하려면 그 파일을 지우고 다시 실행하세요")


def load_resume(path, model, opt, config, fingerprint):
    """resume.pt → 모델·옵티마이저에 싣고 상태 dict를 돌려준다. 설정·데이터가 다르면 멈춘다."""
    try:  # mmap: 1.2B면 14GB인 파일을 통째로 올리지 않고, 싣는 텐서만 읽는다 (sample.load_checkpoint와 같다)
        ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as e:
        if "mmap" not in str(e):
            raise
        ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("config") != config:
        raise SystemExit(f"{path}는 --config {ck.get('config')}로 저장됐습니다 (지금: {config})")
    if ck.get("fingerprint") != fingerprint:
        raise SystemExit(f"{path}: 학습 데이터나 설정이 저장할 때와 다릅니다 — 같은 --data/--extra-data/"
                         "--limit과 같은 설정으로 재개하세요 (다르면 위치가 다른 예제를 가리킵니다)")
    model.load_state_dict(ck.pop("model"))
    opt.load_state_dict(ck.pop("optimizer"))
    # 남은 텐서(DPO의 참조 로그확률)는 복사해 파일에서 떼어 둔다 — 곧 같은 경로에 새 resume.pt를 쓴다
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in ck.items()}


def already_past(stop_after, next_step) -> bool:
    """--stop-after N으로 재개했는데 이미 step N을 마친 상태인가 (그러면 한 step도 더 돌지 않는다)."""
    if stop_after is None or next_step <= stop_after:
        return False
    print(f"--stop-after {stop_after}: 이미 step {next_step - 1}까지 돌았습니다 — 더 돌 것이 없습니다", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="사전학습 체크포인트 (--resume이면 필요 없다)")
    ap.add_argument("--resume", default=None,
                    help="중단된 학습의 resume.pt — 옵티마이저·위치까지 이어서 (데이터 인자·설정은 똑같이)")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="이 step을 마치면 resume.pt를 저장하고 정상 종료 (GPU 양보·실험용)")
    ap.add_argument("--config", default="sft_1b")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--data", default=None, help="JSONL 데이터 (없으면 KoAlpaca + KULLM v2)")
    ap.add_argument("--extra-data", nargs="*", default=[],
                    help="학습에만 더할 JSONL (예: data/math_sft.jsonl). val 분할은 바뀌지 않는다")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--limit", type=int, default=None, help="데이터셋별 예제 수 제한 (테스트용)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if not args.base and not args.resume:
        ap.error("--base(새로 시작) 또는 --resume(이어서)이 필요합니다")

    ckpt_dir = os.path.join(args.out_dir, args.config)
    if not args.resume:
        refuse_stale_resume(ckpt_dir)

    conf = importlib.import_module(f"configs.{args.config}")
    tcfg = conf.train
    device = pick_device(args.device)
    device_type = device_type_of(device)

    tok = ByteBPETokenizer.load(args.tokenizer)
    examples = build_examples(tok, tcfg["max_len"], args.limit, args.data)
    n_val = min(max(64, len(examples) // 100), len(examples) // 5)
    val_ex, train_ex = examples[:n_val], examples[n_val:]
    # 추가 데이터는 분할 '뒤에' 학습 쪽에만 붙인다. 앞에서 섞으면 Random(42) 셔플 결과가
    # 달라져 val이 통째로 바뀌고, 예전 실행과 val loss를 비교할 수 없게 된다.
    for path in args.extra_data:
        extra = build_examples(tok, tcfg["max_len"], None, path)
        train_ex += extra
        print(f"추가 데이터 {path}: {len(extra)}개를 학습에 더함")
    print(f"train {len(train_ex)} / val {len(val_ex)} 예제")

    # CPU로 읽는다. 사전학습 체크포인트엔 옵티마이저 상태(1.2B 기준 약 10GB)도 들어 있어서
    # GPU로 바로 올리면 그만큼이 학습 내내 GPU에 남는다.
    model = GPT(conf.model)
    base = None  # 어느 가중치에서 출발한 학습인가 — resume.pt에 적어 두고 재개할 때 보여준다
    if not args.resume:
        ckpt = load_checkpoint(args.base)
        model.load_state_dict(ckpt["model"])
        base = f"{args.base} (step {ckpt['step']})"
        print(f"베이스 모델: {base}")
        del ckpt
    model = model.to(device)
    opt = torch.optim.AdamW(model.param_groups(tcfg["weight_decay"]),
                            lr=tcfg["max_lr"], betas=(0.9, 0.95), fused=(device_type == "cuda"))

    os.makedirs(ckpt_dir, exist_ok=True)
    pad_id = tok.special_tokens["<|pad|>"]
    B, accum = tcfg["batch_size"], tcfg["grad_accum"]
    group = tcfg.get("group_by_length", True)
    # 길이별 묶음은 덩어리마다 B로 안 나눠지는 나머지를 버리므로 실제 배치 수로 센다
    steps_per_epoch = len(make_batches(train_ex, B, 1000, group)) // accum
    max_steps = steps_per_epoch * tcfg["epochs"]
    assert max_steps > 0, "데이터가 배치 하나 분량도 안 된다"
    # warmup은 절대 step 수(sft_1b: 50)라, 작은 데이터(--data my_chats.jsonl)에서는 전체 step보다
    # 길어져 학습률이 max_lr에 닿지도 못하고 decay도 없이 끝났다. 전체의 10%로 자른다.
    warmup = min(tcfg["warmup_steps"], max(1, max_steps // 10))
    print(f"총 {max_steps} steps ({tcfg['epochs']} epochs, warmup {warmup}) | "
          f"길이별 묶음 {'켬' if group else '끔'}")

    fingerprint = data_fingerprint(train_ex, tcfg)
    start = {"epoch": 0, "step_in_epoch": 0, "step": 0, "bad": 0}
    if args.resume:
        start = load_resume(args.resume, model, opt, args.config, fingerprint)
        base = start.get("base")
        print(f"재개: {args.resume} — epoch {start['epoch']}, step {start['step']}부터 "
              f"(출발한 베이스 모델: {base or '기록 없음'})", flush=True)
        if already_past(args.stop_after, start["step"]):
            return
    else:
        vl0 = val_loss(model, val_ex, pad_id, B, device, device_type, tcfg.get("loss_chunk"))
        print(f"SFT 전 val loss (베이스 모델): {vl0:.4f}", flush=True)

    def checkpoint(epoch, step_in_epoch, step):
        """다음에 돌 자리를 적어 resume.pt로 남긴다."""
        return save_resume(ckpt_dir, model, opt, config=args.config, fingerprint=fingerprint, base=base,
                           epoch=epoch, step_in_epoch=step_in_epoch, step=step, bad=bad)

    model.train()
    step, bad = start["step"], start["bad"]
    t0 = time.time()
    for epoch in range(start["epoch"], tcfg["epochs"]):
        batches = make_batches(train_ex, B, seed=1000 + epoch, group_by_length=group)
        print(f"epoch {epoch}: 패딩 효율 {padding_efficiency(train_ex, batches)*100:.1f}% "
              f"(실제 토큰 / 패딩 포함 토큰)", flush=True)
        first = start["step_in_epoch"] if epoch == start["epoch"] else 0
        for step_in_epoch in range(first, steps_per_epoch):
            # linear warmup → cosine decay
            if step < warmup:
                lr = tcfg["max_lr"] * (step + 1) / warmup
            else:
                frac = (step - warmup) / max(1, max_steps - warmup)
                lr = tcfg["min_lr"] + 0.5 * (tcfg["max_lr"] - tcfg["min_lr"]) * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr

            micro = [collate([train_ex[j] for j in b], pad_id)
                     for b in batches[step_in_epoch * accum : (step_in_epoch + 1) * accum]]
            n_tok = sum(int((y != -100).sum()) for _, y in micro)
            loss_acc = 0.0
            for x, y in micro:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y, loss_chunk=tcfg.get("loss_chunk"))
                # 마이크로배치 평균 × (그 배치 토큰 수 / step 전체 토큰 수) = step 전체의 토큰 평균
                w = int((y != -100).sum()) / max(1, n_tok)
                (loss * w).backward()
                loss_acc += loss.item() * w
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            # 예전엔 검사 없이 opt.step() — NaN 한 번이면 가중치가 전부 NaN이 된 채로 끝까지 돌아
            # latest.pt를 덮어쓰고 '완료'로 끝났다 (train.py에는 있던 가드)
            bad = step_if_finite(opt, loss_acc, gn, bad, step)
            opt.zero_grad(set_to_none=True)

            if step % 10 == 0:
                print(f"epoch {epoch} step {step}/{max_steps} | loss {loss_acc:.4f} | lr {lr:.2e} "
                      f"| {time.time()-t0:.0f}s", flush=True)
            if step > 0 and step % tcfg["save_every"] == 0:
                path = save(model, step, args.config, os.path.join(ckpt_dir, "latest.pt"))
                checkpoint(epoch, step_in_epoch + 1, step + 1)
                print(f"체크포인트 저장: {path} + {RESUME_FILE} (step {step})", flush=True)
            if args.stop_after is not None and step >= args.stop_after:
                path = checkpoint(epoch, step_in_epoch + 1, step + 1)
                print(f"--stop-after {args.stop_after}: step {step}에서 정지 — 이어서 하려면 같은 데이터 "
                      f"인자에 --resume {path}", flush=True)
                return
            step += 1

        vl = val_loss(model, val_ex, pad_id, B, device, device_type, tcfg.get("loss_chunk"))
        print(f"=== epoch {epoch} 종료 | val loss {vl:.4f} ===", flush=True)
        save(model, step - 1, args.config, os.path.join(ckpt_dir, f"epoch{epoch}.pt"))
        path = save(model, step - 1, args.config, os.path.join(ckpt_dir, "latest.pt"))
        if epoch + 1 < tcfg["epochs"]:
            checkpoint(epoch + 1, 0, step)
        print(f"체크포인트 저장: {path}, epoch{epoch}.pt (step {step - 1})", flush=True)

    resume_path = os.path.join(ckpt_dir, RESUME_FILE)
    if os.path.exists(resume_path):  # 끝났으니 재개 상태(옵티마이저 포함, 1.2B면 14GB)는 지운다
        os.remove(resume_path)
    print("SFT 완료 — python sample.py --ckpt", os.path.join(ckpt_dir, "latest.pt"), "--chat")


if __name__ == "__main__":
    main()
