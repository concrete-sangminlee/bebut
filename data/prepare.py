"""코퍼스 스트리밍 → 정제 → 토크나이즈 → shard .bin 생성.

다운로드·정제·토크나이즈를 한 번의 스트리밍 패스로 처리한다 (원문 저장 안 함 —
20B 토큰이면 원문만 ~80GB라 디스크 절약). 결과물:

  data/shards/train_000000.bin, train_000001.bin, ...  (uint16 토큰 id)
  data/shards/val_000000.bin
  data/shards/meta.json                                (토크나이저 경로, shard 크기, 통계)

각 문서는 <|eos|>로 끝난다. 문서 순서는 소스가 섞이도록 라운드로빈. val은 문서 내용의
해시로 고른다(data.clean.is_val) — annealing 준비가 같은 데이터셋을 다시 읽어도 val 문서는
train에 들어가지 않는다. 소스가 예산 전에 바닥나면 알리고, 실제 혼합 비율을 meta.json에 남긴다.
같은 폴더에 shard가 이미 있으면 --overwrite 없이는 멈춘다 (남은 shard가 학습에 섞이지 않게).

사용 예:
  python -m data.prepare --tokens 100e6   # 검증용 0.1B
  python -m data.prepare --tokens 12e9    # 본 학습용 (서버에서, 수일)
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from datasets import load_dataset

from tokenizer.bpe import ByteBPETokenizer

from .clean import VAL_FRACTION, MinHashDeduper, is_val, keep_document

# (HF dataset, config, 비율, 한국어 소스 여부)
MIX = [
    ("HuggingFaceFW/fineweb-2", "kor_Hang", 0.75, True),
    ("wikimedia/wikipedia", "20231101.ko", 0.10, True),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.15, False),
]

SHARD_TOKENS = 50_000_000  # shard당 5천만 토큰 = 100MB(uint16)

_tok = None
_deduper = None


def _init_worker(tok_path):
    global _tok, _deduper
    _tok = ByteBPETokenizer.load(tok_path)
    _deduper = MinHashDeduper()  # 서명 계산 전용 (중복 판정은 메인 프로세스가 담당)


def _process_doc(job):
    """워커: 필터 → MinHash 서명 → 토크나이즈까지 전부 수행.

    중복 판정 자체는 전역 상태(지금까지 본 서명)가 필요해 메인 프로세스가 하지만,
    비싼 연산(해싱·인코딩)은 전부 여기서 병렬로 끝낸다.
    """
    text, korean, val_ratio = job
    if not keep_document(text, korean_source=korean):
        return None, None, False
    keys = _deduper.band_keys(text[:2000])
    return keys, _tok.encode(text) + [_tok.special_tokens["<|eos|>"]], is_val(text, val_ratio)


class ShardWriter:
    def __init__(self, out_dir, split):
        self.out_dir = out_dir
        self.split = split
        self.buf = np.empty(SHARD_TOKENS, dtype=np.uint16)
        self.fill = 0
        self.shard_idx = 0
        self.total = 0

    def add(self, ids):
        arr = np.asarray(ids, dtype=np.uint16)
        self.total += len(arr)
        while len(arr) > 0:
            n = min(len(arr), SHARD_TOKENS - self.fill)
            self.buf[self.fill : self.fill + n] = arr[:n]
            self.fill += n
            arr = arr[n:]
            if self.fill == SHARD_TOKENS:
                self._flush()

    def _flush(self):
        if self.fill == 0:
            return
        path = os.path.join(self.out_dir, f"{self.split}_{self.shard_idx:06d}.bin")
        self.buf[: self.fill].tofile(path)
        print(f"  저장: {path} ({self.fill/1e6:.1f}M tokens)")
        self.shard_idx += 1
        self.fill = 0

    def close(self):
        self._flush()


def clear_old_shards(out_dir, overwrite):
    """이전 실행의 shard가 남아 있으면 멈춘다. 로더는 폴더의 *.bin을 전부 읽으므로, 이번 실행이
    더 적은 shard를 만들면 예전 shard의 꼬리가 그대로 학습에 섞인다. --overwrite면 지운다."""
    import glob

    old = sorted(glob.glob(os.path.join(out_dir, "*_*.bin")))
    if not old:
        return
    if not overwrite:
        raise SystemExit(f"{out_dir}에 이전 shard {len(old)}개가 있습니다 — 다른 폴더를 쓰거나 "
                         "--overwrite로 지우고 다시 만드세요")
    for p in old + [os.path.join(out_dir, "meta.json")]:
        if os.path.exists(p):
            os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=100e6, help="목표 토큰 수")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out-dir", default="data/shards")
    ap.add_argument("--val-ratio", type=float, default=VAL_FRACTION,
                    help="val 문서 비율 (해시로 고른다). data.prepare_anneal과 같은 값을 쓸 것")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--overwrite", action="store_true", help="out-dir의 이전 shard를 지우고 다시 만든다")
    args = ap.parse_args()
    run(args)
    # pyarrow 스레드풀이 인터프리터 종료 시 데드락에 걸리는 문제(macOS)가 있어
    # 모든 산출물을 flush한 뒤 강제 종료한다.
    os._exit(0)


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    clear_old_shards(args.out_dir, args.overwrite)
    target = int(args.tokens)

    streams = []
    for name, config, ratio, korean in MIX:
        ds = load_dataset(name, config, split="train", streaming=True)
        streams.append({"it": iter(ds), "budget": int(target * ratio), "got": 0, "docs": 0,
                        "korean": korean, "name": f"{name}/{config}", "exhausted": False})

    deduper = MinHashDeduper()
    train_w = ShardWriter(args.out_dir, "train")
    val_w = ShardWriter(args.out_dir, "val")
    n_docs = n_filtered = n_dup = 0
    t_start = time.time()

    def doc_stream():
        """소스 라운드로빈으로 원문을 생산 (필터·해싱·인코딩은 워커가 담당)."""
        nonlocal n_docs
        active = list(streams)
        while active:
            for s in list(active):
                if s["got"] >= s["budget"]:
                    active.remove(s)
                    print(f"[{s['name']}] 목표 도달 ({s['got']/1e6:.0f}M tokens)", flush=True)
                    continue
                try:
                    text = next(s["it"])["text"]
                except StopIteration:
                    active.remove(s)
                    s["exhausted"] = True
                    # 예전엔 아무 말 없이 빠졌다 — 위키백과가 260M에서 바닥나 혼합 비율이
                    # 75/10/15가 아니라 82/1/17이 된 것을 아무도 몰랐다
                    print(f"[{s['name']}] 데이터 소진: {s['got']/1e6:.0f}M / 예산 {s['budget']/1e6:.0f}M "
                          f"tokens — 모자란 만큼은 다른 소스로 채워지지 않는다", flush=True)
                    continue
                n_docs += 1
                yield s, text

    with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        # imap의 순서 보존을 이용해 (소스, 결과)를 짝지어 예산을 갱신한다
        pending_sources = []

        def jobs():
            for s, text in doc_stream():
                pending_sources.append(s)
                yield (text, s["korean"], args.val_ratio)

        for keys, ids, val in pool.imap(_process_doc, jobs(), chunksize=16):
            s = pending_sources.pop(0)
            if keys is None:
                n_filtered += 1
                continue
            if deduper.check_keys(keys):
                n_dup += 1
                continue
            s["got"] += len(ids)
            s["docs"] += 1
            # 문서 단위로 train/val 분리 — 해시로 고르므로 다른 스크립트·반복에서도 같은 쪽
            if val:
                val_w.add(ids)
            else:
                train_w.add(ids)
            if (train_w.total + val_w.total) % 10_000_000 < len(ids):
                done = train_w.total + val_w.total
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (target - done) / rate / 3600
                print(f"진행: {done/1e6:.0f}M / {target/1e6:.0f}M tokens "
                      f"(문서 {n_docs}, 필터 {n_filtered}, 중복 {n_dup}) "
                      f"| {rate/1e6:.2f}M tok/s | 남은 시간 ~{eta:.1f}시간", flush=True)

    train_w.close()
    val_w.close()
    meta = {
        "tokenizer": args.tokenizer,
        "shard_tokens": SHARD_TOKENS,
        "train_tokens": train_w.total,
        "val_tokens": val_w.total,
        "docs": n_docs,
        "filtered": n_filtered,
        "duplicates": n_dup,
        "mix": [(m[0], m[1], m[2]) for m in MIX],
        "val_ratio": args.val_ratio, "val_split": "hash",
        "sources": [{"name": s["name"], "budget": s["budget"], "got": s["got"], "docs": s["docs"],
                     "exhausted": s["exhausted"]} for s in streams],
    }
    total = sum(s["got"] for s in streams) or 1
    meta["realized_mix"] = {s["name"]: round(s["got"] / total, 4) for s in streams}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("실제 혼합: " + ", ".join(f"{k} {v:.1%}" for k, v in meta["realized_mix"].items()), flush=True)
    print(f"\n완료: train {train_w.total/1e6:.1f}M / val {val_w.total/1e6:.1f}M tokens", flush=True)
    return meta


if __name__ == "__main__":
    main()
