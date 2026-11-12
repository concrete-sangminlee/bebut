"""Byte-level BPE 토크나이저 — 학습·인코딩·디코딩 전부 직접 구현.

byte-level인 이유: 텍스트를 UTF-8 바이트로 다루면 기본 vocab 256개로 세상 모든
문자를 표현할 수 있어 <unk>가 원천적으로 없다. 한글 음절은 UTF-8에서 3바이트라
처음엔 3토큰이지만, BPE 학습이 자주 나오는 음절·어절을 하나의 토큰으로 merge한다.

학습 알고리즘: 코퍼스를 pretokenize한 뒤 "가장 자주 인접하는 토큰 쌍"을 반복해서
새 토큰으로 합친다. 순진하게 매 step 전체를 다시 세면 O(merges × corpus)라 불가능하고,
여기서는 (1) 동일 pretoken을 빈도와 함께 dedup하고 (2) merge가 건드린 단어의
쌍 빈도만 증분 갱신하며 (3) 최빈 쌍은 lazy-deletion heap으로 꺼낸다.
"""

import heapq
import json
import re
from collections import Counter
from functools import lru_cache

from .pretokenize import pretokenize

SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]


def _count_batch(texts: list[str]) -> Counter:
    """멀티프로세스 워커용: 텍스트 배치의 pretoken 빈도 집계."""
    c: Counter[str] = Counter()
    for t in texts:
        c.update(pretokenize(t))
    return c


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")

class ByteBPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], special_tokens: list[str] | None = None):
        self.merges = merges
        # vocab: id -> bytes. 0-255는 원시 바이트, 이후는 merge 순서대로.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        # merge 우선순위: 학습에서 먼저 만들어진 쌍이 인코딩 때도 먼저 합쳐져야
        # 학습/인코딩 결과가 일치한다.
        self.ranks: dict[tuple[int, int], int] = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens: dict[str, int] = {}
        base = 256 + len(merges)
        for i, tok in enumerate(special_tokens or SPECIAL_TOKENS):
            self.special_tokens[tok] = base + i
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    # ---------------- 인코딩 / 디코딩 ----------------

    def _merge_chunk(self, ids: tuple[int, ...]) -> tuple[int, ...]:
        """한 chunk 안에서 rank가 낮은(=먼저 학습된) 쌍부터 반복 merge."""
        ids = list(ids)
        while len(ids) >= 2:
            best_pair, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_pair, best_rank = pair, r
            if best_pair is None:
                break
            new_id = 256 + best_rank
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return tuple(ids)

    @lru_cache(maxsize=1 << 20)
    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        return self._merge_chunk(tuple(chunk.encode("utf-8")))

    def encode(self, text: str, allowed_special: bool = False) -> list[int]:
        """allowed_special=True면 텍스트 내 special token 문자열을 해당 id로 인코딩.
        학습 데이터의 일반 텍스트에는 False(사용자 입력이 special을 위조 못 하게).

        짝 없는 서로게이트(\\ud83d 등 — JS가 이모지 중간을 자른 문자열을 JSON으로 보내면 생긴다)는
        UTF-8로 인코딩할 수 없어 예전엔 예외로 죽었다(서버는 응답 없이 연결을 끊었다).
        U+FFFD로 바꿔 인코딩한다. 정상 텍스트의 결과는 그대로다.
        """
        try:
            return self._encode(text, allowed_special)
        except UnicodeEncodeError:
            return self._encode(_LONE_SURROGATE.sub("\ufffd", text), allowed_special)

    def _encode(self, text: str, allowed_special: bool) -> list[int]:
        if allowed_special and self.special_tokens:
            import re

            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
        ids: list[int] = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                for chunk in pretokenize(part):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for i in ids:
            if i in self.id_to_special:
                out.extend(self.id_to_special[i].encode("utf-8"))
            else:
                out.extend(self.vocab[i])
        return out.decode("utf-8", errors="replace")

    # ---------------- 학습 ----------------

    @classmethod
    def train(
        cls,
        texts,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        verbose_every: int = 1000,
        workers: int = 1,
    ) -> "ByteBPETokenizer":
        """texts(문자열 iterable)로부터 BPE merge 규칙을 학습한다."""
        special_tokens = special_tokens or SPECIAL_TOKENS
        num_merges = vocab_size - 256 - len(special_tokens)
        assert num_merges > 0

        # 1) pretoken 빈도 집계 — 동일 어절은 한 번만 처리하면 되므로 대폭 절약.
        #    집계는 병렬화 가능 (merge 루프는 순차적이라 불가).
        chunk_counts: Counter[str] = Counter()
        if workers > 1:
            from multiprocessing import Pool

            def batches():
                buf = []
                for t in texts:
                    buf.append(t)
                    if len(buf) >= 256:
                        yield buf
                        buf = []
                if buf:
                    yield buf

            with Pool(workers) as pool:
                for i, c in enumerate(pool.imap_unordered(_count_batch, batches())):
                    chunk_counts.update(c)
                    if i % 1000 == 0:
                        print(f"  집계: {i*256}개 문서, 고유 pretoken {len(chunk_counts):,}", flush=True)
        else:
            for text in texts:
                chunk_counts.update(pretokenize(text))
        print(f"집계 완료: 고유 pretoken {len(chunk_counts):,}개", flush=True)

        # words[i] = (토큰 id 리스트, 코퍼스 내 빈도)
        words: list[list[int]] = []
        counts: list[int] = []
        for chunk, c in chunk_counts.items():
            words.append(list(chunk.encode("utf-8")))
            counts.append(c)
        del chunk_counts

        # 2) 초기 쌍 빈도 + 쌍이 등장하는 단어 인덱스(inverted index)
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_words: dict[tuple[int, int], set[int]] = {}
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += counts[wi]
                pair_to_words.setdefault(pair, set()).add(wi)

        # 3) lazy-deletion max-heap
        heap = [(-c, pair) for pair, c in pair_counts.items()]
        heapq.heapify(heap)
        print(f"merge 루프 시작: 단어 {len(words):,}개, 초기 쌍 {len(pair_counts):,}개", flush=True)

        import time as _time

        t_start = _time.time()
        merges: list[tuple[int, int]] = []
        while len(merges) < num_merges and heap:
            neg_c, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg_c or -neg_c == 0:
                continue  # 낡은 heap 항목
            merges.append(pair)
            new_id = 256 + len(merges) - 1

            # merge가 등장하는 단어만 다시 써서 쌍 빈도를 증분 갱신
            touched = pair_to_words.pop(pair, set())
            pair_counts.pop(pair, None)
            changed: set[tuple[int, int]] = set()
            for wi in touched:
                w = words[wi]
                c = counts[wi]
                # 갱신 전 이 단어가 기여하던 쌍 빈도를 빼고
                for p in zip(w, w[1:]):
                    pair_counts[p] -= c
                    changed.add(p)
                    s = pair_to_words.get(p)
                    if s is not None:
                        s.discard(wi)
                # merge 적용
                new_w, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                        new_w.append(new_id)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                words[wi] = new_w
                # 갱신 후 쌍 빈도를 다시 더한다
                for p in zip(new_w, new_w[1:]):
                    pair_counts[p] += c
                    changed.add(p)
                    pair_to_words.setdefault(p, set()).add(wi)
            for p in changed:
                if pair_counts.get(p, 0) > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))
                else:
                    pair_counts.pop(p, None)
                    pair_to_words.pop(p, None)

            if verbose_every and len(merges) % verbose_every == 0:
                tok = cls._render(merges, pair)
                elapsed = _time.time() - t_start
                rate = len(merges) / elapsed
                eta = (num_merges - len(merges)) / rate / 60
                print(f"merge {len(merges)}/{num_merges}: {tok!r} (빈도 {-neg_c}) "
                      f"| {rate:.1f} merge/s | 남은 시간 ~{eta:.0f}분", flush=True)

        return cls(merges, special_tokens)

    @staticmethod
    def _render(merges: list[tuple[int, int]], pair: tuple[int, int]) -> str:
        vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(merges):
            vocab[256 + i] = vocab[a] + vocab[b]
        return (vocab[pair[0]] + vocab[pair[1]]).decode("utf-8", errors="replace")

    # ---------------- 저장 / 로드 ----------------

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "merges": self.merges,
                    "special_tokens": list(self.special_tokens.keys()),
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "ByteBPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls([tuple(p) for p in data["merges"]], data["special_tokens"])
