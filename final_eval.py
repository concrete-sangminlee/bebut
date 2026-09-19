"""사전학습 완주 후 최종 평가.

annealing 직전(step 19,200)과 최종(latest) 모델을 같은 조건으로 잰다.
  - perplexity: 웹 val(shards_26b)과 고품질 val(shards_anneal) 둘 다.
    annealing 이후 모델은 고품질 분포로 옮겨갔으므로 한쪽만 재면 오해한다.
  - KOBEST: 문항 제한 없이 전체 테스트셋 (마일스톤은 150문항이라 노이즈가 컸다).
    예측 분포도 함께 기록 — 한쪽으로 쏠리면 점수는 라벨 분포일 뿐이다.
  - 사실 정확도: 같은 프롬프트 10회 추출 빈도.

사용: python final_eval.py  →  logs/final_eval.json
"""
import json
import time

import torch

import configs.base_1b as conf
from data.dataloader import ShardedDataLoader
from eval import FORMATTERS, eval_kobest, perplexity
from model import GPT
from tokenizer.bpe import ByteBPETokenizer

DEV = "cuda"
CKPTS = [("annealing 직전", "checkpoints/base_1b/step_019200.pt"),
         ("최종", "checkpoints/base_1b/latest.pt")]
VALS = [("웹", "data/shards_26b"), ("고품질", "data/shards_anneal")]
PROBES = [("대한민국의 수도는", ["서울"]), ("훈민정음을 창제한 왕은", ["세종"]),
          ("물의 화학식은", ["H2O", "H₂O"]), ("한국에서 가장 높은 산은", ["한라산", "백두산"])]

tok = ByteBPETokenizer.load("tokenizer/tokenizer.json")
results = {}
for label, path in CKPTS:
    t0 = time.time()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(conf.model).to(DEV)
    model.load_state_dict(ckpt["model"])
    model.eval()
    r = {"step": ckpt["step"], "ppl": {}, "kobest": {}, "facts": {}}
    print(f"\n===== {label} (step {ckpt['step']}) =====", flush=True)

    for vname, vdir in VALS:
        loader = ShardedDataLoader(vdir, "val", 4, conf.model.max_seq_len, seed=123, device=DEV)
        ppl, loss = perplexity(model, loader, 50, "cuda")
        r["ppl"][vname] = round(ppl, 2)
        print(f"  perplexity [{vname}]: {ppl:.2f}", flush=True)

    for task in FORMATTERS:
        acc, n, preds = eval_kobest(model, tok, task, 2, DEV, None)
        r["kobest"][task] = {"acc": round(acc * 100, 1), "n": n,
                             "preds": {str(k): v for k, v in sorted(preds.items())}}
        print(f"  {task:<10} {acc*100:5.1f}% (n={n}) 예측분포 {dict(sorted(preds.items()))}", flush=True)

    torch.manual_seed(0)
    for prompt, answers in PROBES:
        ids = torch.tensor([tok.encode(prompt)], device=DEV)
        greedy = tok.decode(model.generate(ids, 14, temperature=0.0)[0].tolist())[len(prompt):]
        hits = 0
        for _ in range(10):
            out = tok.decode(model.generate(ids, 12, temperature=0.6, top_p=0.9)[0].tolist())
            hits += any(a in out[len(prompt):] for a in answers)
        r["facts"][prompt] = {"hits": hits, "greedy": greedy.replace("\n", " ").strip()}
        print(f"  [{prompt}] {hits}/10 | greedy: {greedy.strip()[:50]}", flush=True)

    r["seconds"] = round(time.time() - t0)
    results[label] = r
    del model
    torch.cuda.empty_cache()

with open("logs/final_eval.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n저장: logs/final_eval.json", flush=True)
