"""벤치마크 오염 검사 — 평가 문항이 학습 데이터에 들어 있는지 단어 n-gram으로 찾는다.

평가 문항이 학습 데이터에 있으면 점수는 실력이 아니라 기억을 잰다. 이 프로젝트는 이미
SFT 검증 loss에서 그 일을 겪었다(annealing에 SFT 데이터를 넣었기 때문). 여기서는
벤치마크(KMMLU · HAE-RAE · KOBEST)와 자체 문항을 학습 데이터와 대조한다:
SFT(KoAlpaca · KULLM v2), annealing 교과서(maywell/korean_textbooks 17개 구성, mmlu_* 포함),
위키백과(사전학습 10% · annealing 15%), 나무위키(annealing 15%). 사전학습 웹(FineWeb-2)과
영어 FineWeb-Edu는 수백 GB라 대상에 없다.

방법 (GPT-3·Llama 2의 n-gram 오염 검사와 같은 계열):
- 텍스트를 NFKC 정규화 + 소문자로 바꾸고 문자·숫자(한글·한자·라틴·그리스 등)만 남겨
  단어(어절)로 자른다
- 문항마다 단어 8-gram을 색인에 넣는다. 5~7단어짜리 짧은 문항은 문항 전체를 한 덩어리로
  넣어 '통째로 그대로 있을 때'만 걸리게 한다 (5단어 미만은 '검사 불가')
- 정보가 없는 키는 버린다: 두 글자 이상이고 숫자만이 아닌 단어가 4개 미만인 키
  ("1 2 3 4 5 6 7 의", "f x f y f x y f" 같은 수식 조각이 우연히 겹치는 것을 막는다)
- 문항이 여러 필드로 이뤄지면 필드마다 따로 키를 만든다 — 필드 이음새를 가로지르는 키는
  실제 데이터에 나올 수 없는 문자열이라 겹침 비율만 깎는다
- 문항이 여러 표기로 나올 수 있으면(COPA: "전제 그래서 결과", "결과 그래서 전제" …) 표기마다
  키를 만들고 가장 많이 겹친 표기의 비율을 쓴다
- 학습 데이터를 한 번 훑으며 색인에 있는 키를 모은다 (멀티프로세스)
- 겹침 = 문항 키 중 학습 데이터에 나온 비율. 기준(기본 0.5) 이상이면 오염

8-gram을 쓰는 이유: "다음 중 옳지 않은 것은?" 같은 시험 상투구는 4~5어절이라 걸리지
않고, 문항 고유의 문장이 그대로 들어 있을 때만 걸린다. 비율 기준을 두는 것도 한두 개의
흔한 구절로 오염 판정이 나지 않게 하기 위해서다.

BoolQ 지문은 위키백과 문단이라 따로 'boolq_passage'로 잰다. 지문이 학습 데이터에 있는 것은
'지문 노출'이지 문항(질문·정답) 노출이 아니므로 오염 id 목록에는 넣지 않는다.

사용:
  python -m evals.contamination                              # SFT 데이터 대조 (수 분)
  python -m evals.contamination --train koalpaca kullm textbooks wiki namuwiki \\
      --out logs/contamination.json --out-ids evals/contaminated_ids.json
  python -m evals.contamination --train-jsonl my_sft.jsonl   # 직접 만든 데이터
  # 나눠서 훑고 합치기: 출처마다 --save-found, 마지막에 --train 없이 --load-found a.json b.json

결과: 평가 세트별 (문항 수, 검사 가능, 오염, 일부 겹침) + 오염·일부 겹침 문항과 근거(학습 데이터
출처#행) + 오염 문항 id 목록. id 목록은 final_eval.py · eval.py --exclude-ids로 넘겨 오염 문항을
뺀 점수를 낼 수 있다.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from multiprocessing import get_context

N = 8
MIN_WORDS = 5
MIN_CONTENT = 4  # 키 하나에 필요한 '내용 단어'(두 글자 이상, 숫자만이 아님) 수
_WORD = re.compile(r"[^\W_]+")  # 모든 문자 체계의 문자·숫자 (밑줄 제외)
EXPOSURE_ONLY = {"boolq_passage"}  # 노출만 보고하고 오염 id에는 넣지 않는 평가 세트


def words(text: str) -> list[str]:
    """NFKC(전각 → 반각, 호환 자모 → 일반 자모) + 소문자 후 단어 목록."""
    return _WORD.findall(unicodedata.normalize("NFKC", str(text)).lower())


def ngrams(ws: list[str], n: int = N) -> list[str]:
    return [" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def _content(w: str) -> bool:
    return len(w) >= 2 and not w.isdigit()


def item_keys(ws: list[str]) -> list[str]:
    """한 필드의 검사 단위: 8단어 이상이면 8-gram, 5~7단어면 필드 전체 하나, 더 짧으면 없음.
    내용 단어가 MIN_CONTENT개 미만인 키는 버린다. 순서를 지키며 중복을 없앤다."""
    if len(ws) >= N:
        grams = [ws[i:i + N] for i in range(len(ws) - N + 1)]
    elif len(ws) >= MIN_WORDS:
        grams = [ws]
    else:
        return []
    return list(dict.fromkeys(" ".join(g) for g in grams if sum(map(_content, g)) >= MIN_CONTENT))


def form_keys(form) -> list[str]:
    """표기 하나(필드 목록)의 키 — 필드마다 따로 만든다 (이음새를 가로지르지 않는다)."""
    keys = []
    for field in form:
        keys += item_keys(words(field))
    return list(dict.fromkeys(keys))


# ---------------- 평가 세트: (평가 세트, id, 표기 목록) ----------------
# id 형식은 eval.py의 --exclude-ids와 같다: <벤치마크>/<설정>/<test 인덱스>
# 표기 목록: [[필드, ...], ...] — 가장 많이 겹친 표기로 판정한다. 문자열 하나도 받는다.


def copa_forms(row) -> list[list[str]]:
    """COPA 문항이 텍스트에 나올 수 있는 표기들. 전제·답은 대개 2~4어절이라 따로는 검사할 수
    없고, 이어 쓴 문장으로만 잡힌다. 첫 표기는 eval.format_copa가 만드는 채점 문맥과 같다."""
    p, a = row["premise"], row[f"alternative_{row['label'] + 1}"]
    if row["question"] == "원인":  # 전제가 결과, 답이 원인
        return [[f"{p} 왜냐하면 {a}"], [f"{p} {a}"], [f"{a} 그래서 {p}"], [f"{a} {p}"]]
    return [[f"{p} 그래서 {a}"], [f"{p} {a}"]]


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
                item_id = f"kobest/{name}/{i}"
                if name == "copa":
                    yield name, item_id, copa_forms(r)
                elif name == "boolq":  # 질문이 문항, 지문은 위키백과 문단이라 노출만 따로 잰다
                    yield name, item_id, r["question"]
                    yield "boolq_passage", item_id, r["paragraph"]
                else:  # 채점 문맥과 같은 "문맥 정답" — 문맥이 길어 자연스럽게 이어진다
                    yield name, item_id, f"{r['context']} {r['ending_' + str(r['label'] + 1)]}"
        elif name in ("heldout", "math"):
            fn = "heldout_ko.json" if name == "heldout" else "math_heldout_ko.json"
            with open(os.path.join(root, "evals", fn), encoding="utf-8") as f:
                for i, q in enumerate(json.load(f)):
                    yield name, f"{name}/{i}", q["q"]
        else:
            raise ValueError(f"모르는 평가 세트: {name}")


# ---------------- 학습 데이터: (출처, 행 번호, 텍스트) ----------------

TRAIN_SOURCES = {
    "koalpaca": [("beomi/KoAlpaca-v1.1a", None, ("instruction", "output"))],
    "kullm": [("nlpai-lab/kullm-v2", None, ("instruction", "input", "output"))],
    "wiki": [("wikimedia/wikipedia", "20231101.ko", ("text",))],         # 1.4GB
    "namuwiki": [("heegyu/namuwiki-extracted", None, ("text",))],      # 9.7GB
}


def _parts(name):
    if name == "textbooks":
        from data.prepare_anneal import TEXTBOOK_PARTS  # annealing에 실제로 쓴 구성 그대로

        return [(ds, cfg, ("text",)) for ds, cfg in TEXTBOOK_PARTS]
    if name not in TRAIN_SOURCES:
        raise ValueError(f"모르는 학습 데이터: {name}")
    return TRAIN_SOURCES[name]


def iter_train(names, jsonl_paths=(), limit=None):
    from datasets import load_dataset

    for name in names:
        for ds_name, cfg, fields in _parts(name):
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

_KEYS: dict = {}  # 워커 안의 색인: 단어 수 → 키 집합. Pool initializer로 받는다


def _init(keys):
    """워커 초기화. 색인을 인자로 받아야 spawn·forkserver(macOS·Windows 기본, 파이썬 3.14부터
    리눅스 기본)에서도 워커가 빈 색인으로 돌지 않는다. fork면 인자를 복사 없이 물려받는다."""
    global _KEYS
    _KEYS = keys


def _scan(batch):
    """워커: 문서들에서 색인에 있는 키를 찾아 {키: (출처, 행)}로 돌려준다."""
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
    """평가 문항 → (문항별 (표시 텍스트, 표기별 키 목록), 길이별 전체 키 집합).
    문항은 (평가 세트, id)로 구분한다 — BoolQ 질문과 지문이 같은 id를 쓴다."""
    per_item, keys = {}, {}
    for bench, item_id, forms in eval_items:
        if isinstance(forms, str):
            forms = [[forms]]
        fk = [form_keys(f) for f in forms]
        per_item[(bench, item_id)] = (" / ".join(forms[0]), fk)
        for g in (g for ks in fk for g in ks):
            keys.setdefault(g.count(" ") + 1, set()).add(g)
    return per_item, keys


def index_digest(keys) -> str:
    """색인 지문 — 저장해 둔 대조 결과(--save-found)가 같은 색인으로 만든 것인지 확인한다."""
    h = hashlib.sha256()
    for g in sorted(g for ks in keys.values() for g in ks):
        h.update(g.encode() + b"\n")
    return h.hexdigest()[:16]


def scan(train_iter, keys, workers=4, progress_every=200_000, start_method=None):
    """학습 데이터를 훑어 색인에 있는 키와 그 첫 출처를 모은다.
    start_method: None이면 플랫폼 기본 (fork · spawn · forkserver 모두 같은 결과)."""
    frozen = {n: frozenset(k) for n, k in sorted(keys.items(), reverse=True)}
    found, rows, counts, t0 = {}, 0, {}, time.time()

    def counted():
        nonlocal rows
        for src, i, text in train_iter:
            rows += 1
            counts[src] = counts.get(src, 0) + 1
            if rows % progress_every == 0:
                print(f"  {rows:,}행 훑음, 겹친 키 {len(found):,} ({time.time() - t0:.0f}s)",
                      file=sys.stderr, flush=True)
            yield src, i, text

    if workers > 1:
        ctx = get_context(start_method)
        with ctx.Pool(workers, initializer=_init, initargs=(frozen,)) as pool:
            for part in pool.imap_unordered(_scan, _batches(counted()), chunksize=4):
                for g, ev in part.items():
                    found.setdefault(g, ev)
    else:
        _init(frozen)
        for b in _batches(counted()):
            for g, ev in _scan(b).items():
                found.setdefault(g, ev)
    return found, counts


def report(per_item, found, threshold=0.5, examples=5, partial_limit=50):
    """평가 세트별 집계와 오염 문항 id 목록.

    세트마다 items · checkable · flagged · any_match, examples(오염 문항 앞 몇 개),
    partial(일부만 겹친 문항, 겹침 큰 순서로 partial_limit개), sources(겹친 문항 수, 근거 출처별).
    """
    out, flagged_ids = {}, []
    for (bench, item_id), (text, fk) in per_item.items():
        r = out.setdefault(bench, {"items": 0, "checkable": 0, "flagged": 0, "any_match": 0,
                                   "examples": [], "partial": [], "sources": {}})
        r["items"] += 1
        fk = [ks for ks in fk if ks]
        if not fk:
            continue
        r["checkable"] += 1
        overlap, hits = max(((sum(g in found for g in ks) / len(ks), [g for g in ks if g in found])
                             for ks in fk), key=lambda x: x[0])
        if not hits:
            continue
        r["any_match"] += 1
        for s in sorted({found[g][0] for g in hits}):
            r["sources"][s] = r["sources"].get(s, 0) + 1
        src, row = found[hits[0]]
        entry = {"id": item_id, "overlap": round(overlap, 3), "text": text[:160],
                 "evidence": f"{src}#{row}", "match": hits[0]}
        if overlap >= threshold:
            r["flagged"] += 1
            if bench not in EXPOSURE_ONLY:
                flagged_ids.append(item_id)
            if len(r["examples"]) < examples:
                r["examples"].append(entry)
        else:
            r["partial"].append(entry)
    for r in out.values():
        r["partial"] = sorted(r["partial"], key=lambda e: -e["overlap"])[:partial_limit]
        r["sources"] = dict(sorted(r["sources"].items(), key=lambda kv: -kv[1]))
    return out, sorted(flagged_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+",
                    default=["kmmlu", "haerae", "copa", "boolq", "hellaswag", "heldout", "math"])
    ap.add_argument("--train", nargs="*", default=["koalpaca", "kullm"],
                    help="koalpaca · kullm · textbooks(교과서 17개 구성) · wiki(1.4GB) · namuwiki(9.7GB)")
    ap.add_argument("--train-jsonl", nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="출처별 행 수 제한 (빠른 확인용)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--save-found", default=None, help="대조 결과(겹친 키와 근거)를 저장 — 나눠 훑기용")
    ap.add_argument("--load-found", nargs="*", default=[], help="저장해 둔 대조 결과를 합친다")
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    ap.add_argument("--out-ids", default=None, help="오염 문항 id 목록 JSON (--exclude-ids로 넘긴다)")
    args = ap.parse_args()

    t0 = time.time()
    per_item, keys = build_index(iter_evals(args.evals))
    digest = index_digest(keys)
    print(f"평가 문항 {len(per_item):,}개, 키 {sum(len(k) for k in keys.values()):,}개 색인 "
          f"(길이별 {dict(sorted((n, len(k)) for n, k in keys.items()))}, 지문 {digest}, "
          f"{time.time() - t0:.0f}s)", flush=True)
    found, counts = {}, {}
    if args.train or args.train_jsonl:
        found, counts = scan(iter_train(args.train, args.train_jsonl, args.limit), keys, args.workers)
    if args.save_found:
        with open(args.save_found, "w", encoding="utf-8") as f:
            json.dump({"index": digest, "train_rows": counts, "found": found}, f, ensure_ascii=False)
    for path in args.load_found:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved["index"] != digest:
            sys.exit(f"{path}: 다른 색인으로 만든 결과입니다 ({saved['index']} ≠ {digest}) — 같은 "
                     "--evals, 같은 버전의 이 도구로 다시 훑으세요")
        for src, n in saved["train_rows"].items():
            counts[src] = counts.get(src, 0) + n
        for g, ev in saved["found"].items():
            found.setdefault(g, tuple(ev))
    res, flagged = report(per_item, found, args.threshold)
    print(f"학습 데이터 {sum(counts.values()):,}행 ({time.time() - t0:.0f}s)")
    for src, n in counts.items():
        print(f"  {src}: {n:,}행")
    head = f"오염(≥{args.threshold})"
    print(f"\n{'평가 세트':<14} {'문항':>7} {'검사 가능':>9} {head:>10} {'일부 겹침':>9}")
    for bench, r in res.items():
        print(f"{bench:<14} {r['items']:>7,} {r['checkable']:>9,} {r['flagged']:>10,} {r['any_match']:>9,}"
              + ("  (노출만 — 오염 id에 넣지 않음)" if bench in EXPOSURE_ONLY else ""))
        for ex in r["examples"][:3]:
            print(f"    - {ex['id']} 겹침 {ex['overlap']:.0%} ← {ex['evidence']}: {ex['text'][:70]}")
        if r["sources"]:
            print(f"    근거 출처: {r['sources']}")
    summary = {"n": N, "threshold": args.threshold, "index": digest, "train_rows": counts,
               "evals": res, "flagged_total": len(flagged), "seconds": round(time.time() - t0)}
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
