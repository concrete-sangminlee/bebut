"""벤치마크 오염 검사 — 평가 문항이 학습 데이터에 들어 있는지 단어 n-gram으로 찾는다.

평가 문항이 학습 데이터에 있으면 점수는 실력이 아니라 기억을 잰다. 이 프로젝트는 이미
SFT 검증 loss에서 그 일을 겪었다(annealing에 SFT 데이터를 넣었기 때문). 여기서는
벤치마크(KMMLU · HAE-RAE · KOBEST)와 자체 문항을, 실제로 학습에 쓴 데이터 —
SFT(KoAlpaca · KULLM v2)와 annealing 교과서(maywell/korean_textbooks 17개 구성,
mmlu_* 포함) — 와 대조한다.

방법 (GPT-3·Llama 2의 n-gram 오염 검사와 같은 계열):
- 텍스트를 소문자 + 한글·영문·숫자만 남겨 단어(어절)로 자른다
- 문항마다 단어 8-gram을 모두 색인에 넣는다. 5~7단어짜리 짧은 문항은 문항 전체를 한 덩어리로
  넣어 '통째로 그대로 있을 때'만 걸리게 한다 (5단어 미만은 '검사 불가'로 센다)
- 학습 데이터를 한 번 훑으며 색인에 있는 8-gram을 모은다 (멀티프로세스)
- 문항의 8-gram 중 학습 데이터에 나온 비율 = 겹침. 기준(기본 0.5) 이상이면 오염

8-gram을 쓰는 이유: "다음 중 옳지 않은 것은?" 같은 시험 상투구는 4~5어절이라 걸리지
않고, 문항 고유의 문장이 그대로 들어 있을 때만 걸린다. 비율 기준을 두는 것도 한두 개의
흔한 구절로 오염 판정이 나지 않게 하기 위해서다.

사용:
  python -m evals.contamination                              # SFT 데이터 대조 (수 분)
  python -m evals.contamination --train koalpaca kullm textbooks --out logs/contamination.json
  python -m evals.contamination --train-jsonl my_sft.jsonl   # 직접 만든 데이터

결과: 평가 세트별 (문항 수, 검사 가능, 오염, 하나라도 겹침) + 예시 + 오염 문항 id 목록.
id 목록은 final_eval.py --exclude-ids로 넘겨 오염 문항을 뺀 점수를 낼 수 있다.
"""

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool

N = 8
MIN_WORDS = 5
_WORD = re.compile(r"[가-힣a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def ngrams(ws: list[str], n: int = N) -> list[str]:
    return [" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def item_keys(ws: list[str]) -> list[str]:
    """문항의 검사 단위: 8단어 이상이면 8-gram 전부, 5~7단어면 문항 전체 하나, 더 짧으면 없음."""
    if len(ws) >= N:
        return ngrams(ws, N)
    return [" ".join(ws)] if len(ws) >= MIN_WORDS else []


# ---------------- 평가 세트: (id, 텍스트) ----------------
# id 형식은 eval.py의 --exclude-ids와 같다: <벤치마크>/<설정>/<test 인덱스>


def iter_evals(names):
    from datasets import get_dataset_config_names, load_dataset

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in names:
        if name == "kmmlu":
            for sub in get_dataset_config_names("HAERAE-HUB/KMMLU"):
                for i, r in enumerate(load_dataset("HAERAE-HUB/KMMLU", sub, split="test")):
                    yield name, f"kmmlu/{sub}/{i}", r["question"]
        elif name == "haerae":
            for cat in get_dataset_config_names("HAERAE-HUB/HAE_RAE_BENCH_1.0"):
                for i, r in enumerate(load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.0", cat, split="test")):
                    yield name, f"haerae/{cat}/{i}", r["question"]
        elif name in ("copa", "boolq", "hellaswag"):
            for i, r in enumerate(load_dataset("skt/kobest_v1", name, split="test")):
                if name == "copa":  # 전제 + 정답 결과/원인을 이어 쓴 자연스러운 문장
                    text = f"{r['premise']} {r['alternative_' + str(r['label'] + 1)]}"
                elif name == "boolq":
                    text = f"{r['paragraph']} {r['question']}"
                else:
                    text = f"{r['context']} {r['ending_' + str(r['label'] + 1)]}"
                yield name, f"kobest/{name}/{i}", text
        elif name in ("heldout", "math"):
            fn = "heldout_ko.json" if name == "heldout" else "math_heldout_ko.json"
            with open(os.path.join(root, "evals", fn), encoding="utf-8") as f:
                for i, q in enumerate(json.load(f)):
                    yield name, f"{name}/{i}", q["q"]
        else:
            raise ValueError(f"모르는 평가 세트: {name}")


# ---------------- 학습 데이터: (출처, 행 번호, 텍스트) ----------------


def iter_train(names, jsonl_paths=(), limit=None):
    from datasets import load_dataset

    for name in names:
        if name == "koalpaca":
            parts = [("beomi/KoAlpaca-v1.1a", None, ("instruction", "output"))]
        elif name == "kullm":
            parts = [("nlpai-lab/kullm-v2", None, ("instruction", "input", "output"))]
        elif name == "textbooks":
            from data.prepare_anneal import TEXTBOOK_PARTS  # annealing에 실제로 쓴 구성 그대로

            parts = [(ds, cfg, ("text",)) for ds, cfg in TEXTBOOK_PARTS]
        else:
            raise ValueError(f"모르는 학습 데이터: {name}")
        for ds_name, cfg, fields in parts:
            src = ds_name + (f"/{cfg}" if cfg else "")
            rows = load_dataset(ds_name, cfg, split="train", streaming=True)
            for i, r in enumerate(rows):
                if limit and i >= limit:
                    break
                yield src, i, " ".join(str(r.get(f) or "") for f in fields)
    for path in jsonl_paths:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                if "messages" in r:
                    text = " ".join(m.get("content", "") for m in r["messages"])
                else:
                    text = " ".join(str(r.get(k) or "") for k in ("instruction", "input", "output", "text"))
                yield path, i, text


# ---------------- 대조 ----------------

_KEYS: dict = {}  # 단어 수 → 키 집합. fork로 워커에 물려준다 (복사 없이 공유)


def _scan(batch):
    """워커: 문서들에서 색인에 있는 n-gram을 찾아 {n-gram: (출처, 행)}로 돌려준다."""
    found = {}
    for src, i, text in batch:
        ws = words(text)
        for n, keys in _KEYS.items():
            for j in range(len(ws) - n + 1):
                g = " ".join(ws[j:j + n])
                if g in keys and g not in found:
                    found[g] = (src, i)
    return found


def _batches(it, size=256):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def build_index(eval_items):
    """평가 문항 → (문항별 n-gram 목록, 전체 n-gram 집합)."""
    per_item, keys = {}, {}
    for bench, item_id, text in eval_items:
        gs = item_keys(words(text))
        per_item[item_id] = (bench, text, gs)
        for g in gs:
            keys.setdefault(g.count(" ") + 1, set()).add(g)
    return per_item, keys


def scan(train_iter, keys, workers=4, progress_every=200_000):
    """학습 데이터를 훑어 색인에 있는 n-gram과 그 첫 출처를 모은다."""
    global _KEYS
    _KEYS = {n: frozenset(k) for n, k in sorted(keys.items(), reverse=True)}
    found, rows, counts, t0 = {}, 0, {}, time.time()

    def counted():
        nonlocal rows
        for src, i, text in train_iter:
            rows += 1
            counts[src] = counts.get(src, 0) + 1
            if rows % progress_every == 0:
                print(f"  {rows:,}행 훑음, 겹친 n-gram {len(found):,} ({time.time() - t0:.0f}s)",
                      file=sys.stderr, flush=True)
            yield src, i, text

    if workers > 1:
        with Pool(workers) as pool:
            for part in pool.imap_unordered(_scan, _batches(counted()), chunksize=4):
                for g, ev in part.items():
                    found.setdefault(g, ev)
    else:
        for b in _batches(counted()):
            for g, ev in _scan(b).items():
                found.setdefault(g, ev)
    return found, counts


def report(per_item, found, threshold=0.5, examples=5):
    """평가 세트별 집계와 오염 문항 id 목록."""
    out, flagged_ids = {}, []
    for item_id, (bench, text, gs) in per_item.items():
        r = out.setdefault(bench, {"items": 0, "checkable": 0, "flagged": 0, "any_match": 0,
                                   "examples": []})
        r["items"] += 1
        if not gs:
            continue
        r["checkable"] += 1
        hits = [g for g in gs if g in found]
        overlap = len(hits) / len(gs)
        r["any_match"] += bool(hits)
        if overlap >= threshold:
            r["flagged"] += 1
            flagged_ids.append(item_id)
            if len(r["examples"]) < examples:
                src, row = found[hits[0]]
                r["examples"].append({"id": item_id, "overlap": round(overlap, 3),
                                      "text": text[:160], "evidence": f"{src}#{row}"})
    return out, sorted(flagged_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+",
                    default=["kmmlu", "haerae", "copa", "boolq", "hellaswag", "heldout", "math"])
    ap.add_argument("--train", nargs="*", default=["koalpaca", "kullm"],
                    help="koalpaca · kullm · textbooks(annealing 교과서 17개 구성, 수 GB)")
    ap.add_argument("--train-jsonl", nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="출처별 행 수 제한 (빠른 확인용)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    ap.add_argument("--out-ids", default=None, help="오염 문항 id 목록 JSON (final_eval.py --exclude-ids)")
    args = ap.parse_args()

    t0 = time.time()
    per_item, keys = build_index(iter_evals(args.evals))
    print(f"평가 문항 {len(per_item):,}개, 키 {sum(len(k) for k in keys.values()):,}개 색인 "
          f"(길이별 {dict(sorted((n, len(k)) for n, k in keys.items()))}, {time.time() - t0:.0f}s)", flush=True)
    found, counts = scan(iter_train(args.train, args.train_jsonl, args.limit), keys, args.workers)
    res, flagged = report(per_item, found, args.threshold)
    print(f"학습 데이터 {sum(counts.values()):,}행 ({time.time() - t0:.0f}s)")
    for src, n in counts.items():
        print(f"  {src}: {n:,}행")
    head = f"오염(≥{args.threshold})"
    print(f"\n{'평가 세트':<10} {'문항':>7} {'검사 가능':>9} {head:>10} {'일부 겹침':>9}")
    for bench, r in res.items():
        print(f"{bench:<10} {r['items']:>7,} {r['checkable']:>9,} {r['flagged']:>10,} {r['any_match']:>9,}")
        for ex in r["examples"][:3]:
            print(f"    - {ex['id']} 겹침 {ex['overlap']:.0%} ← {ex['evidence']}: {ex['text'][:70]}")
    summary = {"n": N, "threshold": args.threshold, "train_rows": counts, "evals": res,
               "flagged_total": len(flagged), "seconds": round(time.time() - t0)}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"\n저장: {args.out}")
    if args.out_ids:
        with open(args.out_ids, "w", encoding="utf-8") as f:
            json.dump(flagged, f, ensure_ascii=False, indent=0)
        print(f"오염 문항 id {len(flagged)}개: {args.out_ids}")


if __name__ == "__main__":
    main()
