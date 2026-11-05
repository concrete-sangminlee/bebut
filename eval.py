"""평가: held-out perplexity + KOBEST few-shot.

perplexity: val shard에서 exp(평균 next-token loss). 낮을수록 좋다.

KOBEST(BoolQ·COPA·HellaSwag): 베이스 모델은 지시를 못 따르므로 log-likelihood
스코어링을 쓴다 — 문제를 텍스트로 만들고 각 선택지를 이어붙여 점수를 매긴다.
점수는 비조건부 정규화를 거친다(normalized_score 참고): 로그확률 합만 쓰면 짧거나
흔한 선택지가 문맥과 무관하게 이긴다. 예측 분포도 함께 반환해 쏠림을 감지한다.

2026-11 수정: 선택지 첫 토큰이 채점에서 빠지던 버그를 고쳤다 (choice_span 참고).
그 전에 잰 KOBEST 수치(logs/의 final_eval.json, sft_eval.json, milestones)는 전부
이 버그의 영향을 받았다.

사용 예:
  python eval.py --ckpt checkpoints/base_1b/latest.pt --ppl
  python eval.py --ckpt checkpoints/base_1b/latest.pt --kobest copa --shots 5
"""

import argparse
import math

import torch
import torch.nn.functional as F

from data.dataloader import ShardedDataLoader
from sample import load_model, pick_device


@torch.no_grad()
def perplexity(model, loader, iters, device_type, loss_chunk=8192):
    """val shard에서 exp(평균 loss). 청크 CE를 써서 64K vocab logits를 통째로 만들지 않는다."""
    losses = []
    for _ in range(iters):
        x, y = loader.next_batch()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y, loss_chunk=loss_chunk)
        losses.append(loss.item())
    mean = sum(losses) / len(losses)
    return torch.exp(torch.tensor(mean)).item(), mean


def token_byte_lengths(tok, vocab_size: int, device) -> torch.Tensor:
    """토큰 id → UTF-8 바이트 수. special token은 0 (텍스트가 아니라서 BPB에서 뺀다)."""
    lens = torch.zeros(vocab_size, dtype=torch.long)
    for i, b in tok.vocab.items():
        if i < vocab_size:
            lens[i] = len(b)
    return lens.to(device)


@torch.no_grad()
def bits_per_byte(model, loader, iters, device_type, tok, chunk=8192):
    """val shard의 bits-per-byte = 전체 손실(비트) / 원문 바이트 수.

    perplexity는 '토큰당' 지표라 토크나이저가 다르면 비교할 수 없다. 64K vocab은 토큰
    하나에 한국어 2~3글자를 담으므로 32K vocab 모델보다 perplexity가 높게 나오는 게
    정상이다. 바이트당으로 바꾸면 어휘 크기·분할 방식과 무관해져 다른 모델과 나란히 놓을
    수 있다. special token(<|eos|> 등)은 바이트가 없으므로 분자·분모에서 모두 뺀다.
    """
    nbytes = token_byte_lengths(tok, model.cfg.vocab_size, next(model.parameters()).device)
    total_nats, total_bytes = 0.0, 0
    for _ in range(iters):
        x, y = loader.next_batch()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            h = model.hidden_states(x)
        h, y = h.reshape(-1, h.size(-1)), y.reshape(-1)
        for i in range(0, h.size(0), chunk):  # 64K vocab logits를 한 번에 만들지 않는다
            logits = model.lm_head(h[i:i + chunk]).float()
            nll = F.cross_entropy(logits, y[i:i + chunk], reduction="none")
            b = nbytes[y[i:i + chunk]]
            total_nats += float((nll * (b > 0)).sum())
            total_bytes += int(b.sum())
    return total_nats / (math.log(2) * max(1, total_bytes))


def choice_span(tok, context: str, choice: str) -> tuple[list[int], int]:
    """context + choice를 통째로 토큰화하고, choice를 덮는 첫 토큰의 위치를 찾는다.

    **context와 choice를 따로 토큰화해서 길이로 경계를 잡으면 틀린다.** 문맥이 공백으로
    끝나면(\"답: \", \"그래서 \") pretokenizer가 그 공백을 선택지 첫 단어에 붙이기 때문이다:

        encode("답: ")    → [..., ":", " "]
        encode("답: 예")  → [..., ":", " 예"]      ← 공백이 '예'에 붙었다

    예전 구현은 len(encode(context))를 경계로 써서 **선택지 첫 토큰을 항상 빠뜨렸다.**
    BoolQ의 '예'는 한 토큰이라 채점할 토큰이 0개(점수가 늘 0)였고, '아니오'는 '오'
    하나만 채점됐다. '정규화 전 150문항 전부 예', '정규화 후 98% 아니오'라는 기록된
    증상이 둘 다 이것으로 설명된다 — BoolQ는 모델을 측정한 적이 없었다.

    여기서는 전체 토큰의 바이트 오프셋을 누적해, 문맥 안에 완전히 들어가는 토큰까지를
    문맥으로, choice에 한 바이트라도 걸치는 토큰부터를 채점 대상으로 삼는다.
    """
    ids = tok.encode(context + choice)
    ctx_bytes = len(context.encode("utf-8"))
    end, start = 0, 0
    for i, t in enumerate(ids):
        end += len(tok.vocab[t])
        if end <= ctx_bytes:
            start = i + 1
        else:
            break
    if start == 0:
        raise ValueError("문맥이 비어 있으면 첫 토큰의 조건부 확률을 정의할 수 없다")
    return ids, start


@torch.no_grad()
def score_continuations(model, tok, pairs: list[tuple[str, str]], device) -> list[float]:
    """(context, choice) 쌍들의 log P(choice | context)를 한 번의 forward로 계산한다.

    오른쪽 패딩으로 묶고, lm_head는 채점에 필요한 위치에만 적용한다 —
    64K vocab의 전 위치 logits(문항당 수백 MB)를 만들 필요가 없다.
    """
    spans = [choice_span(tok, c, ch) for c, ch in pairs]
    L = max(len(ids) for ids, _ in spans) - 1
    x = torch.zeros(len(spans), L, dtype=torch.long)
    rows, pos, tgt = [], [], []
    for b, (ids, start) in enumerate(spans):
        x[b, : len(ids) - 1] = torch.tensor(ids[:-1])
        for t in range(start, len(ids)):  # 위치 t-1의 출력이 토큰 t를 예측한다
            rows.append(b)
            pos.append(t - 1)
            tgt.append(ids[t])
    h = model.hidden_states(x.to(device))
    sel = h[torch.tensor(rows, device=device), torch.tensor(pos, device=device)]
    logp = F.log_softmax(model.lm_head(sel).float(), dim=-1)
    tok_lp = logp.gather(1, torch.tensor(tgt, device=device)[:, None])[:, 0]
    out = torch.zeros(len(spans), dtype=torch.float32, device=device)
    out.index_add_(0, torch.tensor(rows, device=device), tok_lp)
    return out.tolist()


def choice_logprob(model, tok, context: str, choice: str, device) -> tuple[float, int]:
    """context 뒤에 choice가 이어질 로그확률 합과 채점된 토큰 수."""
    ids, start = choice_span(tok, context, choice)
    return score_continuations(model, tok, [(context, choice)], device)[0], len(ids) - start


UNCOND_CONTEXT = "답: "


def normalized_score(model, tok, context: str, choice: str, device) -> float:
    """GPT-3 논문 방식의 비조건부 정규화 점수.

    로그확률 합만 쓰면 '원래 흔한 표현'이 문맥과 무관하게 이긴다. 그래서 문맥 없이
    계산한 로그확률을 뺀다:
        score = log P(choice | context) - log P(choice | 중립 문맥)
    이러면 '문맥이 이 선택지의 확률을 얼마나 끌어올렸는가'만 남는다.
    선택지 길이가 다를 때 생기는 편향도 함께 상쇄된다.
    """
    cond, uncond = score_continuations(
        model, tok, [(context, choice), (UNCOND_CONTEXT, choice)], device)
    return cond - uncond


def format_copa(row):
    conn = "왜냐하면" if row["question"] == "원인" else "그래서"
    ctx = f"{row['premise']} {conn} "
    return ctx, [row["alternative_1"], row["alternative_2"]], row["label"]


def format_boolq(row):
    ctx = f"{row['paragraph']}\n질문: {row['question']}\n답: "
    return ctx, ["아니오", "예"], row["label"]


def format_hellaswag(row):
    return row["context"] + " ", [row[f"ending_{i}"] for i in range(1, 5)], row["label"]


FORMATTERS = {"copa": format_copa, "boolq": format_boolq, "hellaswag": format_hellaswag}


def evaluate_choices(model, tok, fmt, train_rows, test_rows, shots, device):
    """다지선다 채점. 세 가지 점수를 한 번에 낸다.

    acc        log P(choice | context) 합이 가장 큰 선택지 (원시 점수)
    acc_len    위 값을 선택지 바이트 수로 나눈 것 (lm-eval-harness의 acc_norm)
    acc_calib  비조건부 정규화 (normalized_score) — 이 저장소의 기본 지표

    예측 분포도 지표별로 센다. 한쪽으로 쏠리면 정확도는 라벨 분포일 뿐이다.
    """
    prefix = ""
    for i in range(shots):  # few-shot 예시는 train 앞쪽에서 고정 (재현성)
        ctx, choices, label = fmt(train_rows[i])
        prefix += ctx + choices[label] + "\n\n"

    uncond_cache: dict[str, float] = {}  # BoolQ의 '예/아니오'처럼 반복되는 선택지는 한 번만
    names = ("acc", "acc_len", "acc_calib")
    correct = dict.fromkeys(names, 0)
    preds = {n: {} for n in names}
    total = 0
    for row in test_rows:
        ctx, choices, label = fmt(row)
        cond = score_continuations(model, tok, [(prefix + ctx, c) for c in choices], device)
        todo = [c for c in dict.fromkeys(choices) if c not in uncond_cache]
        if todo:
            for c, s in zip(todo, score_continuations(
                    model, tok, [(UNCOND_CONTEXT, c) for c in todo], device)):
                uncond_cache[c] = s
        scores = {
            "acc": cond,
            "acc_len": [s / max(1, len(c.encode("utf-8"))) for s, c in zip(cond, choices)],
            "acc_calib": [s - uncond_cache[c] for s, c in zip(cond, choices)],
        }
        for n in names:
            p = max(range(len(choices)), key=scores[n].__getitem__)
            preds[n][p] = preds[n].get(p, 0) + 1
            correct[n] += int(p == label)
        total += 1
    return {"n": total, **{n: correct[n] / max(1, total) for n in names},
            "preds": {n: dict(sorted(v.items())) for n, v in preds.items()}}


def load_kobest(task, limit=None):
    from datasets import load_dataset

    ds = load_dataset("skt/kobest_v1", task)
    test = ds["test"] if limit is None else ds["test"].select(range(min(limit, len(ds["test"]))))
    return ds["train"], test


def kobest_details(model, tok, task, shots, device, limit=None) -> dict:
    train, test = load_kobest(task, limit)
    return evaluate_choices(model, tok, FORMATTERS[task], train, test, shots, device)


def eval_kobest(model, tok, task, shots, device, limit=None):
    """(정확도, 문항 수, 예측 분포) — 기존 호출부 호환용. 정확도는 acc_calib."""
    r = kobest_details(model, tok, task, shots, device, limit)
    return r["acc_calib"], r["n"], r["preds"]["acc_calib"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--ppl", action="store_true")
    ap.add_argument("--shard-dir", default="data/shards_26b")
    ap.add_argument("--kobest", choices=list(FORMATTERS), default=None)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = pick_device()
    device_type = "cuda" if device == "cuda" else "cpu" if device == "cpu" else "mps"
    model, tok = load_model(args.ckpt, args.tokenizer, device, torch.float32)  # 채점은 fp32로

    if args.ppl:
        loader = ShardedDataLoader(args.shard_dir, "val", 8, model.cfg.max_seq_len,
                                   seed=123, device=device)
        ppl, loss = perplexity(model, loader, 20, device_type)
        print(f"val perplexity: {ppl:.2f} (loss {loss:.4f})")
        loader.step = 0  # 같은 배치로
        print(f"val bits-per-byte: {bits_per_byte(model, loader, 20, device_type, tok):.4f} "
              "(토크나이저와 무관 — 다른 모델과 비교 가능)")

    if args.kobest:
        r = kobest_details(model, tok, args.kobest, args.shots, device, args.limit)
        chance = {"copa": 0.5, "boolq": 0.5, "hellaswag": 0.25}[args.kobest]
        print(f"KOBEST {args.kobest} ({args.shots}-shot, n={r['n']}, "
              f"무작위 기준 {chance*100:.0f}%)")
        # 한쪽으로 쏠려 있으면 정확도가 라벨 분포를 반영할 뿐 성능이 아니다
        for name in ("acc_calib", "acc", "acc_len"):
            print(f"  {name:<9} {r[name]*100:5.1f}%  예측 분포 {r['preds'][name]}")


if __name__ == "__main__":
    main()
