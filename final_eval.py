"""사전학습 완주 후 최종 평가.

annealing 직전(step 19,200)과 최종(latest) 모델을 같은 조건으로 잰다.
  - perplexity·bits-per-byte: 웹 val(shards_26b)과 고품질 val(shards_anneal) 둘 다.
    BPB는 토크나이저와 무관해 다른 한국어 모델과 비교할 수 있다.
    annealing 이후 모델은 고품질 분포로 옮겨갔으므로 한쪽만 재면 오해한다.
  - KOBEST: 문항 제한 없이 전체 테스트셋 (마일스톤은 150문항이라 노이즈가 컸다).
  - KMMLU(45과목, 5-shot)·HAE-RAE Bench(6범주): 다른 한국어 모델과 비교하기 위한 표준
    벤치마크. 기본 cloze 형식, --format mc면 리더보드와 같은 보기·문자 형식.
    acc_calib·acc·acc_len과 지표별 예측 분포를 모두 기록 — 한쪽으로 쏠리면 점수는
    라벨 분포일 뿐이다.
  - 사실 정확도: 같은 프롬프트 10회 추출 빈도.

사용:
  python final_eval.py                                → logs/final_eval.json
  python final_eval.py --ckpts 최종=checkpoints/base_1b/latest.pt --out logs/final_eval_v2.json
"""
import argparse
import json
import time

import torch

from data.dataloader import ShardedDataLoader
from eval import BENCHMARKS, bits_per_byte, perplexity, run_benchmarks
from sample import load_model, pick_device

DEFAULT_CKPTS = ["annealing 직전=checkpoints/base_1b/step_019200.pt",
                 "최종=checkpoints/base_1b/latest.pt"]
VALS = [("웹", "data/shards_26b"), ("고품질", "data/shards_anneal")]
PROBES = [("대한민국의 수도는", ["서울"]), ("훈민정음을 창제한 왕은", ["세종"]),
          ("물의 화학식은", ["H2O", "H₂O"]), ("한국에서 가장 높은 산은", ["한라산", "백두산"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS, help="라벨=체크포인트 경로")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--shots", type=int, default=2, help="KOBEST few-shot (예전 평가와 같은 2)")
    ap.add_argument("--benches", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    ap.add_argument("--bench-shots", type=int, default=5, help="KMMLU·HAE-RAE few-shot")
    ap.add_argument("--format", choices=["cloze", "mc"], default="cloze")
    ap.add_argument("--limit", type=int, default=None, help="과제(과목)별 문항 수 제한")
    ap.add_argument("--skip-ppl", action="store_true", help="shard가 없는 환경에서")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="logs/final_eval.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    device_type = "cuda" if device == "cuda" else "cpu"
    results = {}
    for spec in args.ckpts:
        label, path = spec.split("=", 1)
        t0 = time.time()
        model, tok = load_model(path, args.tokenizer, device, torch.float32)
        step = model.meta.get("step")
        r = {"step": step, "ppl": {}, "bpb": {}, "kobest": {}, "bench": {}, "facts": {}}
        print(f"\n===== {label} (step {step}) =====", flush=True)

        if not args.skip_ppl:
            for vname, vdir in VALS:
                loader = ShardedDataLoader(vdir, "val", 4, model.cfg.max_seq_len, seed=123,
                                           device=device)
                ppl, _ = perplexity(model, loader, 50, device_type)
                loader.step = 0  # 같은 배치로 BPB도
                bpb = bits_per_byte(model, loader, 50, device_type, tok)
                r["ppl"][vname] = round(ppl, 2)
                r["bpb"][vname] = round(bpb, 4)
                print(f"  perplexity [{vname}]: {ppl:.2f} | bits-per-byte {bpb:.4f}", flush=True)

        r["kobest"], r["bench"] = run_benchmarks(model, tok, args.benches, device, args.shots,
                                                 args.bench_shots, args.format, args.limit)

        gen = torch.Generator(device=device).manual_seed(0)
        for prompt, answers in PROBES:
            ids = torch.tensor([tok.encode(prompt)], device=device)
            greedy = tok.decode(model.generate(ids, 14, temperature=0.0)[0].tolist())[len(prompt):]
            hits = 0
            for _ in range(10):
                out = tok.decode(model.generate(ids, 12, temperature=0.6, top_p=0.9,
                                                generator=gen)[0].tolist())
                hits += any(a in out[len(prompt):] for a in answers)
            r["facts"][prompt] = {"hits": hits, "greedy": greedy.replace("\n", " ").strip()}
            print(f"  [{prompt}] {hits}/10 | greedy: {greedy.strip()[:50]}", flush=True)

        r["seconds"] = round(time.time() - t0)
        results[label] = r
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {args.out}", flush=True)


if __name__ == "__main__":
    main()
