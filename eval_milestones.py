"""중간 스냅샷들을 평가해 '능력이 언제 생기는가' 곡선을 그린다.

학습 중 GPU를 방해하지 않도록 CPU에서 돌린다 (CUDA를 아예 안 보이게 한다).
KOBEST는 문항당 선택지 수만큼 forward가 필요해 느리므로 --limit로 표본을 줄인다.

사용: python eval_milestones.py --config base_1b --limit 500
결과: checkpoints/<config>/milestones.csv   step별 요약 (그래프용)
      checkpoints/<config>/milestones.json  문항별 정오까지 — 두 step을 같은 문항끼리 짝 비교한다:
          python compare_evals.py checkpoints/base_1b/milestones.json --a "step 4000" --b "step 16000"
      예전엔 CSV의 백분율만 남아서, 단계 사이의 변화가 표본 오차보다 큰지 볼 방법이 없었다 (RESULTS §1).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU 학습 방해 방지

import argparse
import csv
import glob
import json
import re
import time

import torch

torch.set_num_threads(int(os.environ.get("EVAL_THREADS", "24")))

import importlib

from eval import FORMATTERS, kobest_details, perplexity
from data.dataloader import ShardedDataLoader
from model import GPT
from sample import load_checkpoint
from tokenizer.bpe import ByteBPETokenizer


def save_details(path, label, record):
    """milestones.json에 한 step의 결과를 더한다 (원자적으로 — 평가 도중에 죽어도 파일이 깨지지 않게)."""
    doc = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    doc[label] = record
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base_1b")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--shard-dir", default="data/shards_26b")
    ap.add_argument("--limit", type=int, default=200, help="KOBEST 문항 수 (속도 위해 제한)")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--ppl-iters", type=int, default=8)
    args = ap.parse_args()

    ckpt_dir = os.path.join("checkpoints", args.config)
    snaps = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if not snaps:
        print(f"{ckpt_dir}에 스냅샷(step_*.pt)이 없다. milestone_every 설정을 확인할 것.")
        return

    conf = importlib.import_module(f"configs.{args.config}")
    tok = ByteBPETokenizer.load(args.tokenizer)
    out_path = os.path.join(ckpt_dir, "milestones.csv")

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {int(r["step"]) for r in csv.DictReader(f) if r.get("step")}
    write_header = not os.path.exists(out_path)

    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["step", "tokens", "ppl", "copa", "boolq", "hellaswag", "seconds"])

        for path in snaps:
            step = int(re.search(r"step_(\d+)", path).group(1))
            if step in done:
                continue
            t0 = time.time()
            try:
                ckpt = load_checkpoint(path)
            except Exception as e:  # 쓰는 중이거나 잘린 파일 — 예전엔 여기서 죽어 뒤의 스냅샷도 못 봤다
                print(f"건너뜀: {path}를 읽지 못했다 ({type(e).__name__}) — 다음 실행 때 다시 시도")
                continue
            model = GPT(conf.model)
            model.load_state_dict(ckpt["model"])
            model.eval()

            loader = ShardedDataLoader(args.shard_dir, "val", 2, conf.model.max_seq_len,
                                       seed=123, device="cpu")
            ppl, _ = perplexity(model, loader, args.ppl_iters, "cpu")

            scores, details = {}, {}
            for task in FORMATTERS:
                details[task] = kobest_details(model, tok, task, args.shots, "cpu", args.limit)
                scores[task] = details[task]["acc_calib"]
                # 예측이 한 선택지로 쏠리면 정확도는 라벨 분포일 뿐이다
                preds = details[task]["preds"]["acc_calib"]
                top = max(preds.values()) / sum(preds.values())
                if top > 0.9:
                    print(f"    경고: {task} 예측이 한쪽으로 쏠림 {dict(preds)}", flush=True)

            tokens = step * conf.train["batch_size"] * conf.train["grad_accum"] * conf.model.max_seq_len
            save_details(os.path.join(ckpt_dir, "milestones.json"), f"step {step}",
                         {"step": step, "tokens": tokens, "ppl": round(ppl, 2), "shots": args.shots,
                          "limit": args.limit, "kobest": details})
            row = [step, tokens, f"{ppl:.2f}",
                   f"{scores['copa']*100:.1f}", f"{scores['boolq']*100:.1f}",
                   f"{scores['hellaswag']*100:.1f}", f"{time.time()-t0:.0f}"]
            w.writerow(row)
            f.flush()
            print(f"step {step:6d} ({tokens/1e9:.1f}B tokens) | ppl {ppl:6.2f} | "
                  f"COPA {scores['copa']*100:.1f}% BoolQ {scores['boolq']*100:.1f}% "
                  f"HellaSwag {scores['hellaswag']*100:.1f}% | {time.time()-t0:.0f}초", flush=True)

    print(f"\n결과: {out_path}")
    print("무작위 기준: COPA 50% / BoolQ 50% / HellaSwag 25%")


if __name__ == "__main__":
    main()
