"""BPE merge를 연결 리스트 + 힙으로 바꾼 뒤에도 예전 구현과 토큰이 같은지, 긴 chunk에서 빨라졌는지."""

import random
import time

from tokenizer.bpe import ByteBPETokenizer


def merge_reference(ranks, ids):
    """예전 _merge_chunk 그대로 — merge마다 chunk 전체를 훑는다."""
    ids = list(ids)
    while len(ids) >= 2:
        best_pair, best_rank = None, None
        for pair in zip(ids, ids[1:]):
            r = ranks.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_pair, best_rank = pair, r
        if best_pair is None:
            break
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                out.append(256 + best_rank)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return tuple(ids)


def test_heap_merge_equals_reference(real_tok):
    rng = random.Random(0)
    pools = ["abcdefghijklmnopqrstuvwxyz", "이다는을를에의가하고서도로한지있어기리사대수인시자정", "aaaa", "ab",
             "!@#$%^&*()-_=+", "가각갂", "theofandininingtion하였다있습니다", "0123456789"]
    for _ in range(3000):
        s = "".join(rng.choice(rng.choice(pools)) for _ in range(rng.randint(0, 60)))
        b = tuple(s.encode())
        assert real_tok._merge_chunk(b) == merge_reference(real_tok.ranks, b), s
    for _ in range(1000):  # 유효하지 않은 UTF-8 바이트열도
        b = tuple(rng.randrange(256) for _ in range(rng.randint(0, 40)))
        assert real_tok._merge_chunk(b) == merge_reference(real_tok.ranks, b)


def test_long_chunk_is_not_quadratic(real_tok):
    rng = random.Random(1)
    s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(20000))  # 공백 없는 한 chunk
    b = tuple(s.encode())
    t0 = time.perf_counter()
    fast = real_tok._merge_chunk(b)
    t_fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = merge_reference(real_tok.ranks, b)
    t_ref = time.perf_counter() - t0
    assert fast == ref and t_fast * 10 < t_ref  # 예전 구현은 20,000자에서 1초 남짓, 새 구현은 수십 ms
    assert ByteBPETokenizer.decode(real_tok, list(fast)) == s
