"""Annealing(학습률 decay) 구간용 고품질 데이터 준비.

사전학습 본 구간은 웹 데이터가 주력이지만, 학습률이 0으로 수렴하는 마지막
구간에서 본 데이터는 거의 그대로 모델에 남는다. 그래서 이 구간에는 지식 밀도가
높은 데이터를 넣는다 (Llama 3, MiniCPM, OLMo 2가 쓰는 방식).

혼합 비율은 각 코퍼스를 품질 분류기로 실측한 결과를 근거로 정했다:
    위키백과   중앙값 0.986   교과서 0.995~0.998   나무위키 0.400
    FineWeb-2 ko 0.009        FineWeb-Edu(영어) 0.023

품질 필터는 **한국어 웹과 나무위키에만** 건다. 분류기가 한국어 positive로
학습돼서 영어는 품질과 무관하게 낮은 점수를 받기 때문이다 (영어에 걸면 99.8%가
버려진다). 위키·교과서는 이미 positive 클래스라 거를 이유가 없다.

사용: python -m data.prepare_anneal --tokens 5e9 --out-dir data/shards_anneal
"""

import argparse
import json
import os
import time
import unicodedata
from collections import Counter
from multiprocessing import Pool


from chat import render
from tokenizer.bpe import ByteBPETokenizer

from .clean import VAL_FRACTION, MinHashDeduper, is_val, keep_document
from .prepare import SHARD_TOKENS, ShardWriter, clear_old_shards
from .quality import QualityClassifier

# maywell/korean_textbooks 중 실제로 데이터가 있는 구성만.
# 나머지 mmlu_* 구성들은 업스트림이 깨져 있다 (필드명이 '0'이고 내용이 빈 문자열).
TEXTBOOK_PARTS = [("maywell/korean_textbooks", c) for c in [
    "ko_wikidata", "tiny-textbooks", "claude_evol", "normal_instructions",
    "helpsteer", "code-alpaca",
    "mmlu_all", "mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy",
    "mmlu_business_ethics", "mmlu_clinical_knowledge", "mmlu_college_biology",
    "mmlu_college_chemistry", "mmlu_college_computer_science",
    "mmlu_college_mathematics", "mmlu_college_medicine",
]]

# 비율은 data.measure_sources 실측 + 1차 준비 실행의 실제 산출량을 근거로 정했다.
# 1회분 가용량: 위키백과 260M / 나무위키 540M / 교과서 1,300M / instruction 67M.
#   (위키백과는 스트리밍 앞부분에 긴 문서가 몰려 있어 표본 추정이 5배 부풀려진다.
#    1차 실행에서 실제로 나온 260M을 쓴다.)
# repeat는 예산에 도달하면 자동으로 멈추므로 넉넉히 준다.
# dedup: 웹·나무위키에만 적용. 정제된 코퍼스는 의도적으로 반복 투입하므로 끈다.
SOURCES = [
    dict(label="교과서", parts=TEXTBOOK_PARTS,
         ratio=0.33, threshold=None, repeat=3, korean=True, dedup=False),
    dict(label="위키백과", parts=[("wikimedia/wikipedia", "20231101.ko")],
         ratio=0.15, threshold=None, repeat=5, korean=True, dedup=False),
    dict(label="나무위키", parts=[("heegyu/namuwiki-extracted", None)],
         ratio=0.15, threshold=0.3, repeat=2, korean=True, dedup=True),
    dict(label="선별 웹", parts=[("HuggingFaceFW/fineweb-2", "kor_Hang")],
         ratio=0.23, threshold=0.3, repeat=1, korean=True, dedup=True),
    dict(label="영어 Edu", parts=[("HuggingFaceFW/fineweb-edu", "sample-10BT")],
         ratio=0.10, threshold=None, repeat=1, korean=False, dedup=True),
    dict(label="instruction", parts=[("nlpai-lab/kullm-v2", None),
                                     ("beomi/KoAlpaca-v1.1a", None)],
         ratio=0.04, threshold=None, repeat=4, korean=True, dedup=False,
         instruction=True),
]

_tok = None
_qc = None
_deduper = None


def _init_worker(tok_path, qc_path):
    global _tok, _qc, _deduper
    _tok = ByteBPETokenizer.load(tok_path)
    _qc = QualityClassifier.load(qc_path)
    _deduper = MinHashDeduper()


def _process(job):
    """워커: 필터 → 품질 점수 → (필요시) MinHash 서명 → 토크나이즈.

    dedup=False인 소스는 서명을 만들지 않는다. 위키·교과서처럼 이미 정제된
    코퍼스는 내부 중복이 없고, 오히려 **의도적으로 여러 번 반복해서 넣는다**.
    여기에 중복 제거를 걸면 2회차부터 전부 버려져 반복이 무효가 된다
    (2026-08-13에 실제로 발생: 위키 3회 반복이 1회분만 남았다).
    """
    text, threshold, korean, is_instruction, dedup, val_ratio = job
    text = unicodedata.normalize("NFC", text)  # NFD 한글은 한글 비율 0으로 버려졌다
    val = is_val(text, val_ratio)
    if is_instruction:
        # SFT와 같은 chat 형식으로 넣는다 — 이후 SFT 효과가 좋아진다는 게 최신 관행
        ids = _tok.encode(text, allowed_special=True) + [_tok.special_tokens["<|eos|>"]]
        return (_deduper.band_keys(text[:2000]) if dedup else []), ids, val
    if not keep_document(text, korean_source=korean):
        return None, None, False
    if threshold is not None and _qc.score(text) < threshold:
        return None, None, False
    keys = _deduper.band_keys(text[:2000]) if dedup else []
    return keys, _tok.encode(text) + [_tok.special_tokens["<|eos|>"]], val


def build_stream(src):
    """한 소스의 모든 part를 repeat만큼 순회하며 (회차, 작업 단위)를 생산.

    한 소스가 여러 데이터셋/구성(parts)으로 이뤄질 수 있다. 예: 교과서는 17개 구성,
    instruction은 두 데이터셋. 예산에 도달하면 호출 측에서 중단하므로
    repeat는 넉넉히 잡아도 된다.
    """
    from .prepare import load_dataset  # 늦은 import (테스트가 바꿔 끼운다)

    for pass_idx in range(src["repeat"]):
        for name, config in src["parts"]:
            try:
                ds = load_dataset(name, config, split="train", streaming=True)
            except Exception as e:
                print(f"  [{src['label']}] {name}/{config} 열기 실패: "
                      f"{type(e).__name__}", flush=True)
                continue
            for row in ds:
                if src.get("instruction"):
                    user = row.get("instruction") or ""
                    if row.get("input"):
                        user += "\n\n" + row["input"]
                    ans = row.get("output") or ""
                    if not user or not ans:
                        continue
                    # chat.py와 같은 형식 (문자열로 넣고 allowed_special로 인코딩)
                    text = render([{"role": "user", "content": user},
                                   {"role": "assistant", "content": ans}],
                                  add_generation_prompt=False).removesuffix("\n")
                else:
                    text = row.get("text")
                if text:
                    yield pass_idx, (text, src["threshold"], src["korean"],
                                     bool(src.get("instruction")), src.get("dedup", True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=5e9)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--quality", default="data/quality_ko.npz")
    ap.add_argument("--out-dir", default="data/shards_anneal")
    ap.add_argument("--val-ratio", type=float, default=VAL_FRACTION,
                    help="val 문서 비율 (해시로 고른다). data.prepare와 같은 값이어야 사전학습 val이 "
                         "annealing train에 들어가지 않는다")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--overwrite", action="store_true", help="out-dir의 이전 shard를 지우고 다시 만든다")
    args = ap.parse_args()
    run(args)
    os._exit(0)  # pyarrow 스레드풀 종료 데드락 우회 (prepare.py와 동일)


def run(args):
    """val은 문서 해시로 고르고(data.clean.is_val), 반복 회차 2회차부터는 val 문서를 건너뛴다.
    예전에는 처음 나온 문서 N토큰을 val로 써서, dedup 없이 반복하는 교과서·위키·instruction의
    val 문서가 2회차부터 train에 들어갔다(val perplexity가 학습한 글로 재졌다).

    dedup=True이고 repeat>1인 소스(나무위키)는 1회차에서만 중복을 판정하고, 2회차부터는 1회차에
    채택된 문서만 다시 넣는다. 예전에는 2회차 문서가 1회차와 '중복'으로 전부 버려져 반복이
    무효였다(나무위키 2회 반복 → 1회분 540M만 남음).
    """
    os.makedirs(args.out_dir, exist_ok=True)
    clear_old_shards(args.out_dir, args.overwrite)
    target = int(args.tokens)

    streams = []
    for src in SOURCES:
        budget = int(target * src["ratio"])
        streams.append({**src, "budget": budget, "got": 0, "it": build_stream(src), "seen": 0,
                        "kept": 0, "val_skipped": 0, "kept_once": Counter(), "this_pass": (0, Counter()),
                        "exhausted": False})

    deduper = MinHashDeduper()
    train_w = ShardWriter(args.out_dir, "train")
    val_w = ShardWriter(args.out_dir, "val")
    n_docs = n_filtered = n_dup = 0
    t_start = time.time()

    def jobs():
        """소스 라운드로빈. 예산을 채운 소스는 빠진다."""
        nonlocal n_docs
        active = list(streams)
        while active:
            for s in list(active):
                if s["got"] >= s["budget"]:
                    active.remove(s)
                    print(f"[{s['label']}] 목표 도달 {s['got']/1e6:.0f}M tokens "
                          f"(문서 {s['seen']} 중 {s['kept']} 채택)", flush=True)
                    continue
                try:
                    pass_idx, job = next(s["it"])
                except StopIteration:
                    active.remove(s)
                    s["exhausted"] = True
                    print(f"[{s['label']}] 소진 {s['got']/1e6:.0f}M / 예산 {s['budget']/1e6:.0f}M tokens "
                          f"(문서 {s['seen']} 중 {s['kept']} 채택)", flush=True)
                    continue
                s["seen"] += 1
                n_docs += 1
                yield s, pass_idx, job

    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.tokenizer, args.quality)) as pool:
        pending = []

        def payloads():
            for s, pass_idx, job in jobs():
                pending.append((s, pass_idx))
                yield (*job, args.val_ratio)

        for keys, ids, val in pool.imap(_process, payloads(), chunksize=8):
            s, pass_idx = pending.pop(0)
            if keys is None:
                n_filtered += 1
                continue
            if val and pass_idx > 0:  # val 문서는 1회차에 한 번만 val로
                s["val_skipped"] += 1
                continue
            if keys:  # keys가 비면 dedup 대상이 아니다
                sig = b"".join(keys)
                if pass_idx == 0:
                    if deduper.check_keys(keys):
                        n_dup += 1
                        continue
                    if s["repeat"] > 1:
                        s["kept_once"][sig] += 1
                else:  # 2회차부터는 1회차에 채택된 문서만, 채택된 횟수까지만 (정확한 사본은 서명이 같다)
                    if s["this_pass"][0] != pass_idx:
                        s["this_pass"] = (pass_idx, Counter())
                    used = s["this_pass"][1]
                    if used[sig] >= s["kept_once"][sig]:
                        n_dup += 1
                        continue
                    used[sig] += 1
            s["got"] += len(ids)
            s["kept"] += 1
            (val_w if val else train_w).add(ids)
            done = train_w.total + val_w.total
            if done % 10_000_000 < len(ids):
                rate = done / (time.time() - t_start)
                print(f"진행: {done/1e6:.0f}M / {target/1e6:.0f}M tokens "
                      f"(문서 {n_docs}, 필터 {n_filtered}, 중복 {n_dup}) "
                      f"| {rate/1e6:.2f}M tok/s | 남은 시간 ~{(target-done)/rate/3600:.1f}시간",
                      flush=True)

    train_w.close()
    val_w.close()
    short = [s for s in streams if s["got"] < 0.98 * s["budget"]]
    for s in short:
        print(f"경고: [{s['label']}] 예산의 {s['got'] / max(1, s['budget']):.0%}만 채움 "
              f"({s['got']/1e6:.0f}M / {s['budget']/1e6:.0f}M) — 혼합 비율이 설정과 다르다", flush=True)
    total = sum(s["got"] for s in streams) or 1
    meta = {
        "purpose": "annealing (LR decay 구간용 고품질 혼합)",
        "tokenizer": args.tokenizer,
        "quality_model": args.quality,
        "shard_tokens": SHARD_TOKENS,
        "train_tokens": train_w.total,
        "val_tokens": val_w.total,
        "val_ratio": args.val_ratio, "val_split": "hash",
        "docs": n_docs, "filtered": n_filtered, "duplicates": n_dup,
        "sources": [{"label": s["label"], "parts": s["parts"], "ratio": s["ratio"],
                     "threshold": s["threshold"], "repeat": s["repeat"],
                     "dedup": s.get("dedup", True), "budget": s["budget"], "got": s["got"],
                     "seen": s["seen"], "kept": s["kept"], "val_skipped": s["val_skipped"],
                     "exhausted": s["exhausted"]} for s in streams],
        "realized_mix": {s["label"]: round(s["got"] / total, 4) for s in streams},
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("실제 혼합: " + ", ".join(f"{k} {v:.1%}" for k, v in meta["realized_mix"].items()), flush=True)
    print(f"\n완료: train {train_w.total/1e6:.1f}M / val {val_w.total/1e6:.1f}M tokens", flush=True)
    return meta


if __name__ == "__main__":
    main()
