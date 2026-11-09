"""SFT 평가 — 베이스 / SFT 에폭0 / SFT 최종을 같은 조건으로 비교.

1) 대화 평가: evals/heldout_ko.json (학습 데이터와 겹치지 않음을 check_contamination.py로 확인).
   greedy 생성 후 정답 키워드 포함 여부로 채점. '오염대조'는 학습 데이터에 거의 같은 문제가
   있는 것이라 채점에서 빼고 따로 보고한다 — 안 본 문제와의 차이가 기억 효과다.
   정상 종료율(<|im_end|>로 스스로 멈췄는가)과 반복률도 잰다.
2) KOBEST 전체 문항 (사전학습 최종 평가와 같은 2-shot 형식) — SFT가 기존 능력을 깎았는지 본다.
   acc_calib(기본 지표)·acc·acc_len을 모두 기록한다.

사용:
  python sft_eval.py                                   # 기본 세 모델 → logs/sft_eval.json
  python sft_eval.py --models 최종=checkpoints/sft_1b/latest.pt --repetition-penalty 1.1
  python sft_eval.py --skip-kobest                     # 대화 평가만 (빠름)
  python sft_eval.py --benches copa boolq hellaswag kmmlu haerae   # 표준 벤치마크까지
"""
import argparse
import json
import re
import time

import torch

from chat import encode_prompt
from eval import BENCHMARKS, load_exclude, run_benchmarks
from sample import load_model, pick_device

DEFAULT_MODELS = ["베이스=checkpoints/base_1b/latest.pt",
                  "SFT 에폭0=checkpoints/sft_1b/epoch0.pt",
                  "SFT 최종=checkpoints/sft_1b/latest.pt"]


def correct(ans, keys, match=None):
    """정답 키워드가 답에 있으면 정답. match="last_number"면 답의 마지막 숫자만 본다 —
    산술 풀이(47 + 38 = 70 + 15 = 85)엔 중간 숫자가 많아, 아무 데나 있으면 맞다고 치면
    틀린 최종 답도 통과할 수 있다."""
    if match == "last_number":
        nums = re.findall(r"-?\d[\d,]*", ans)
        return bool(nums) and nums[-1].replace(",", "") in keys
    for k in keys:
        if k.isdigit():  # 숫자는 '17' 안의 '7'처럼 걸리지 않게 경계를 본다
            if re.search(rf"(?<!\d){k}(?!\d)", ans):
                return True
        elif k in ans:
            return True
    return False


def repetitive(ans):
    """8자 이상 구절이 3번 이상 반복되면 반복 루프로 본다."""
    s = re.sub(r"\s+", " ", ans)
    for L in (8, 12, 20):
        seen = {}
        for i in range(len(s) - L + 1):
            seen[s[i:i + L]] = seen.get(s[i:i + L], 0) + 1
            if seen[s[i:i + L]] >= 3:
                return True
    return False


def chat_eval(model, tok, questions, max_new, repetition_penalty=1.0):
    end = tok.special_tokens["<|im_end|>"]
    device = model.tok_emb.weight.device
    by_type, stopped, rep, rows = {}, 0, 0, []
    for q in questions:
        ids = encode_prompt(tok, [{"role": "user", "content": q["q"]}])
        out = model.generate(torch.tensor([ids], device=device), max_new, temperature=0.0,
                             eos_id=end, repetition_penalty=repetition_penalty)
        gen = out[0, len(ids):].tolist()
        ended = end in gen
        ans = tok.decode([t for t in gen if t != end]).strip()
        stopped += ended
        r = repetitive(ans)
        rep += r
        ok = correct(ans, q["a"], q.get("match")) if q["a"] else None
        if ok is not None:
            by_type.setdefault(q["type"], []).append(ok)
        rows.append({"type": q["type"], "q": q["q"], "answer": ans, "correct": ok,
                     "stopped": ended, "repetitive": r})
    acc = {t: round(100 * sum(v) / len(v), 1) for t, v in by_type.items()}
    held = [ok for t, v in by_type.items() if t != "오염대조" for ok in v]
    return {"heldout_acc": round(100 * sum(held) / max(1, len(held)), 1), "by_type": acc,
            "stop_rate": round(100 * stopped / len(questions), 1),
            "repeat_rate": round(100 * rep / len(questions), 1), "answers": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="라벨=체크포인트 경로")
    ap.add_argument("--questions", default="evals/heldout_ko.json")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--skip-kobest", action="store_true", help="벤치마크를 모두 건너뛴다")
    ap.add_argument("--benches", nargs="+", choices=BENCHMARKS, default=["copa", "boolq", "hellaswag"],
                    help="kmmlu·haerae도 넣을 수 있다")
    ap.add_argument("--bench-shots", type=int, default=5)
    ap.add_argument("--format", choices=["cloze", "mc"], default="cloze")
    ap.add_argument("--exclude-ids", default=None,
                    help="오염 문항 id 목록 JSON (evals/contamination.py --out-ids) — 빼고 채점")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="logs/sft_eval.json")
    ap.add_argument("--answers-out", default="logs/sft_eval_answers.md")
    args = ap.parse_args()

    device = pick_device(args.device)
    questions = json.load(open(args.questions, encoding="utf-8"))
    results, answers_md = {}, ["# SFT 평가 — 모델별 답변\n"]
    for spec in args.models:
        label, path = spec.split("=", 1)
        t0 = time.time()
        # 채점이 dtype에 흔들리지 않게 fp32로 잰다 (예전 평가와 같은 조건)
        model, tok = load_model(path, args.tokenizer, device, torch.float32)
        print(f"\n===== {label} ({path}) =====", flush=True)
        r = chat_eval(model, tok, questions, args.max_new, args.repetition_penalty)
        print(f"  대화 정답률(비오염) {r['heldout_acc']}% | 유형별 {r['by_type']}", flush=True)
        print(f"  스스로 멈춤 {r['stop_rate']}% | 반복 루프 {r['repeat_rate']}%", flush=True)
        answers_md.append(f"\n## {label}\n")
        for row in r["answers"]:
            mark = "" if row["correct"] is None else ("O " if row["correct"] else "X ")
            answers_md.append(f"- **[{row['type']}] {row['q']}** {mark}\n"
                              f"  > {row['answer'][:300].replace(chr(10), ' ')}\n")

        r["kobest"], r["bench"] = {}, {}
        if not args.skip_kobest:
            r["kobest"], r["bench"] = run_benchmarks(model, tok, args.benches, device, args.shots,
                                                     args.bench_shots, args.format, args.limit,
                                                 load_exclude(args.exclude_ids))

        r["seconds"] = round(time.time() - t0)
        results[label] = r
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(args.answers_out, "w", encoding="utf-8").write("\n".join(answers_md))
    print(f"\n저장: {args.out}, {args.answers_out}", flush=True)


if __name__ == "__main__":
    main()
