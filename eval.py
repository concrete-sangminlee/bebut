"""평가: held-out perplexity + KOBEST few-shot.

perplexity: val shard에서 exp(평균 next-token loss). 낮을수록 좋다.

KOBEST(BoolQ·COPA·HellaSwag): 베이스 모델은 지시를 못 따르므로 log-likelihood
스코어링을 쓴다 — 문제를 텍스트로 만들고 각 선택지를 이어붙여 점수를 매긴다.
점수는 비조건부 정규화를 거친다(normalized_score 참고): 로그확률 합만 쓰면 짧거나
흔한 선택지가 문맥과 무관하게 이긴다. 예측 분포도 함께 반환해 쏠림을 감지한다.

사용 예:
  python eval.py --ckpt checkpoints/base_1b/latest.pt --ppl
  python eval.py --ckpt checkpoints/base_1b/latest.pt --kobest copa --shots 5
"""

import argparse

import torch
import torch.nn.functional as F

from data.dataloader import ShardedDataLoader
from sample import load_model


@torch.no_grad()
def perplexity(model, loader, iters, device_type):
    losses = []
    for _ in range(iters):
        x, y = loader.next_batch()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    mean = sum(losses) / len(losses)
    return torch.exp(torch.tensor(mean)).item(), mean


@torch.no_grad()
def choice_logprob(model, tok, context: str, choice: str, device) -> tuple[float, int]:
    """context 뒤에 choice가 이어질 로그확률 합과 choice의 토큰 수."""
    ctx_ids = tok.encode(context)
    full_ids = tok.encode(context + choice)
    x = torch.tensor([full_ids[:-1]], device=device)
    y = full_ids[1:]
    logits, _ = model(x, targets=torch.tensor([y], device=device))
    logp = F.log_softmax(logits[0].float(), dim=-1)
    start = len(ctx_ids) - 1  # y에서 choice 첫 토큰의 위치
    total = sum(logp[t, y[t]].item() for t in range(start, len(y)))
    return total, max(1, len(y) - start)


def normalized_score(model, tok, context: str, choice: str, device) -> float:
    """GPT-3 논문 방식의 비조건부 정규화 점수.

    로그확률 합만 쓰면 '원래 흔한 표현'이 문맥과 무관하게 이긴다. 실제로 우리
    BoolQ 평가에서 모델이 150문항 전부 '예'를 골라 점수가 라벨 분포(53.3%)에
    고정되는 일이 있었다 ('예'가 '아니오'보다 훨씬 흔한 표현이라서).

    그래서 문맥 없이 계산한 로그확률로 나눈다:
        score = log P(choice | context) - log P(choice | 중립 문맥)
    이러면 '문맥이 이 선택지의 확률을 얼마나 끌어올렸는가'만 남는다.
    선택지 길이가 다를 때 생기는 편향도 함께 상쇄된다.
    """
    cond, _ = choice_logprob(model, tok, context, choice, device)
    uncond, _ = choice_logprob(model, tok, "답: ", choice, device)
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


def eval_kobest(model, tok, task, shots, device, limit=None):
    from datasets import load_dataset

    ds = load_dataset("skt/kobest_v1", task)
    train, test = ds["train"], ds["test"]
    fmt = FORMATTERS[task]

    # few-shot 예시는 train 앞쪽에서 고정으로 뽑는다 (재현성)
    prefix = ""
    for i in range(shots):
        ctx, choices, label = fmt(train[i])
        prefix += ctx + choices[label] + "\n\n"

    correct = total = 0
    pred_counts = {}  # 예측 쏠림 감지용 — 점수만 보면 한쪽으로만 답해도 알 수 없다
    rows = test if limit is None else test.select(range(min(limit, len(test))))
    for row in rows:
        ctx, choices, label = fmt(row)
        scores = [normalized_score(model, tok, prefix + ctx, c, device) for c in choices]
        pred = max(range(len(scores)), key=scores.__getitem__)
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
        correct += int(pred == label)
        total += 1
    return correct / total, total, pred_counts


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

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu" if device == "cpu" else "mps"
    model, tok = load_model(args.ckpt, args.tokenizer, device)

    if args.ppl:
        loader = ShardedDataLoader(args.shard_dir, "val", 8, model.cfg.max_seq_len,
                                   seed=123, device=device)
        ppl, loss = perplexity(model, loader, 20, device_type)
        print(f"val perplexity: {ppl:.2f} (loss {loss:.4f})")

    if args.kobest:
        acc, n, preds = eval_kobest(model, tok, args.kobest, args.shots, device, args.limit)
        chance = {"copa": 0.5, "boolq": 0.5, "hellaswag": 0.25}[args.kobest]
        print(f"KOBEST {args.kobest} ({args.shots}-shot, n={n}): "
              f"정확도 {acc*100:.1f}% (무작위 기준 {chance*100:.0f}%)")
        # 한쪽으로 쏠려 있으면 정확도가 라벨 분포를 반영할 뿐 성능이 아니다
        print(f"  예측 분포: {dict(sorted(preds.items()))}")


if __name__ == "__main__":
    main()
