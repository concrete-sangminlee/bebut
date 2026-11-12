"""SFT 평가 — 베이스 / SFT 에폭0 / SFT 최종을 같은 조건으로 비교.

1) 대화 평가: evals/heldout_ko.json (SFT 데이터와 겹치지 않음을 evals/check_contamination.py로,
   사전학습까지 포함한 학습 데이터 전체는 evals/contamination.py로 확인 — 흔한 질문 하나(heldout/1)가
   웹에 그대로 있어 --exclude-ids를 주면 빼고 채점한다).
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
import os
import re
import time

import torch

from chat import encode_prompt
from eval import BENCHMARKS, load_exclude, run_benchmarks
from sample import load_model, pick_device

DEFAULT_MODELS = ["베이스=checkpoints/base_1b/latest.pt",
                  "SFT 에폭0=checkpoints/sft_1b/epoch0.pt",
                  "SFT 최종=checkpoints/sft_1b/latest.pt"]


def correct(ans, keys, match=None, question=None):
    """정답 키워드가 답에 있으면 정답. match="last_number"면 답의 마지막 숫자만 본다 —
    산술 풀이(47 + 38 = 70 + 15 = 85)엔 중간 숫자가 많아, 아무 데나 있으면 맞다고 치면
    틀린 최종 답도 통과할 수 있다.

    숫자로 시작하거나 끝나는 키는 숫자 경계를 본다 — '3개'가 '13개'에, '4년'이 '14년'에 걸리지
    않게(예전엔 키 전체가 숫자일 때만 봤다). question을 주면 질문에서 따옴표로 묶은 부분(보기
    목록 등)을 답에서 지우고 본다 — "'가지, 바나나, 사과' 중 과일이 아닌 것은?"에 '사과'라고
    틀리게 답해도 질문을 따라 쓴 '가지'가 남아 정답으로 쳤다(커밋된 결과의 세 모델 모두).
    """
    if match == "last_number":
        nums = re.findall(r"-?\d[\d,]*", ans)
        return bool(nums) and nums[-1].replace(",", "") in keys
    if question:
        for quoted in re.findall(r"['‘\"“]([^'’\"”]+)['’\"”]", question):
            ans = ans.replace(quoted, " ")
    for k in keys:
        pat = re.escape(k)
        if k[:1].isdigit():
            pat = r"(?<!\d)" + pat
        if k[-1:].isdigit():
            pat += r"(?!\d)"
        if re.search(pat, ans):
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


def chat_eval(model, tok, questions, max_new, repetition_penalty=1.0, excluded=()):
    """excluded: 채점에서 뺄 문항 번호(evals/contamination.py가 학습 데이터에서 찾은 것). 빼되
    따로 정답률을 남긴다. 멈춤은 대화 CLI·서버와 같이 <|im_end|>와 <|eos|> 둘 다로 본다
    (예전엔 <|im_end|>만 봐서 <|eos|> 뒤까지 생성해 채점했다)."""
    stops = [tok.special_tokens["<|im_end|>"], tok.special_tokens["<|eos|>"]]
    device = model.tok_emb.weight.device
    rows = []
    for q in questions:
        ids = encode_prompt(tok, [{"role": "user", "content": q["q"]}])
        out = model.generate(torch.tensor([ids], device=device), max_new, temperature=0.0,
                             eos_id=stops, repetition_penalty=repetition_penalty)
        gen = out[0, len(ids):].tolist()
        cut = next((j for j, t in enumerate(gen) if t in stops), None)
        ans = tok.decode(gen[:cut]).strip()
        rows.append({"type": q["type"], "q": q["q"], "answer": ans, "stopped": cut is not None,
                     "repetitive": repetitive(ans)})
    return rescore({"answers": rows}, questions, excluded)


def rescore(result, questions, excluded=()):
    """답변 목록을 지금의 채점 규칙으로 채점한다. 저장된 결과(logs/sft_eval.json)에도 쓴다 — 모델 없이."""
    by_type, ex, rows = {}, [], result["answers"]
    for i, (q, row) in enumerate(zip(questions, rows)):
        ok = correct(row["answer"], q["a"], q.get("match"), q["q"]) if q["a"] else None
        row.update(correct=ok, excluded=i in excluded)
        if ok is not None:
            (ex if i in excluded else by_type.setdefault(q["type"], [])).append(ok)
    acc = {t: round(100 * sum(v) / len(v), 1) for t, v in by_type.items()}
    held = [ok for t, v in by_type.items() if t != "오염대조" for ok in v]
    n = max(1, len(rows))
    return {"heldout_acc": round(100 * sum(held) / max(1, len(held)), 1), "by_type": acc,
            "n_scored": len(held), "excluded": len(ex),
            "excluded_acc": round(100 * sum(ex) / len(ex), 1) if ex else None,
            "stop_rate": round(100 * sum(r["stopped"] for r in rows) / n, 1),
            "repeat_rate": round(100 * sum(r["repetitive"] for r in rows) / n, 1), "answers": rows}


def excluded_items(exclude, questions_path):
    """오염 id 집합 → 이 문항 파일의 문항 번호 집합 (heldout_ko.json → heldout/i, math → math/i)."""
    if not exclude:
        return set()
    prefix = "math/" if "math" in os.path.basename(questions_path) else "heldout/"
    return {int(e[len(prefix):]) for e in exclude if e.startswith(prefix) and e[len(prefix):].isdigit()}


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
                    help="오염 문항 id 목록 JSON (evals/contamination.py --out-ids) — 대화 평가와 "
                         "벤치마크 모두 빼고 채점하고, 뺀 문항은 따로 정답률을 남긴다")
    ap.add_argument("--rescore", default=None,
                    help="모델 없이, 저장된 결과 JSON(logs/sft_eval.json)의 답변을 지금 규칙으로 다시 채점")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="logs/sft_eval.json")
    ap.add_argument("--answers-out", default="logs/sft_eval_answers.md")
    args = ap.parse_args()

    device = pick_device(args.device)
    questions = json.load(open(args.questions, encoding="utf-8"))
    exclude = load_exclude(args.exclude_ids)
    if args.rescore:  # 모델 없이 저장된 답변만 다시 채점
        saved = json.load(open(args.rescore, encoding="utf-8"))
        for label, r in saved.items():
            r.update(rescore(r, questions, excluded_items(exclude, args.questions)))
            print(f"{label}: 대화 정답률 {r['heldout_acc']}% (채점 {r['n_scored']}문항, 오염 제외 "
                  f"{r['excluded']}) | 유형별 {r['by_type']}")
        json.dump(saved, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"저장: {args.out}")
        return
    results, answers_md = {}, ["# SFT 평가 — 모델별 답변\n"]
    for spec in args.models:
        label, path = spec.split("=", 1)
        t0 = time.time()
        # 채점이 dtype에 흔들리지 않게 fp32로 잰다 (예전 평가와 같은 조건)
        model, tok = load_model(path, args.tokenizer, device, torch.float32)
        print(f"\n===== {label} ({path}) =====", flush=True)
        r = chat_eval(model, tok, questions, args.max_new, args.repetition_penalty,
                      excluded_items(exclude, args.questions))
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
                                                     args.bench_shots, args.format, args.limit, exclude)

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
