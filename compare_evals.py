"""두 평가 결과를 비교한다 — 차이, 95% 신뢰구간, 유의성.

"A가 B보다 x%p 높다"는 그 차이가 잡음보다 클 때만 의미가 있다. 같은 문항으로 잰 두 결과에
문항별 정오가 들어 있으면(이 저장소의 평가 스크립트가 남긴다) 짝 비교(McNemar)를 하고,
정답 수만 있는 예전 기록이면 짝 없는 비교를 한다. 방법은 evals/stats.py.

사용:
  python compare_evals.py logs/sft_eval.json --a 베이스 --b "SFT 최종"
  python compare_evals.py logs/final_eval.json --a "annealing 직전" --b 최종
  python compare_evals.py run1.json run2.json --a 최종 --b 최종 --metric acc      # 파일 두 개
"""

import argparse
import json

from evals.stats import compare_results, format_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="결과 JSON 하나(두 모델이 한 파일에) 또는 둘(A, B 순서)")
    ap.add_argument("--a", required=True, help="기준 모델의 라벨")
    ap.add_argument("--b", required=True, help="비교할 모델의 라벨")
    ap.add_argument("--metric", choices=["acc_calib", "acc", "acc_len"], default="acc_calib",
                    help="벤치마크 지표 (예전 기록은 지표가 하나라 무시된다)")
    ap.add_argument("--out", default=None, help="비교 결과 JSON")
    args = ap.parse_args()
    if len(args.files) > 2:
        ap.error("파일은 하나 또는 둘")

    docs = [json.load(open(f, encoding="utf-8")) for f in args.files]
    for doc, label, f in ((docs[0], args.a, args.files[0]), (docs[-1], args.b, args.files[-1])):
        if label not in doc:
            ap.error(f"{f}에 '{label}'이 없습니다 (있는 라벨: {', '.join(doc)})")
    res = compare_results(docs[0][args.a], docs[-1][args.b], args.metric)
    print(f"A = {args.a}  →  B = {args.b}")
    for name, c in res.items():
        print(format_comparison(name, c))
    if not res:
        print("  비교할 공통 항목이 없습니다")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"a": args.a, "b": args.b, "metric": args.metric, "comparisons": res}, f,
                      ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
