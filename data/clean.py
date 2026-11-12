"""문서 정제 필터.

FineWeb-2는 이미 강하게 정제된 코퍼스라(언어 필터, dedup, 품질 필터 적용됨)
여기서는 가벼운 안전망만 둔다: 길이, 한글 비율(한국어 소스에만), 반복 비율.
MinHash 근사 중복 제거는 위키 등 소스 간 중복 제거용.
"""

import hashlib
import re

_hangul = re.compile(r"[가-힣]")


def hangul_ratio(text: str) -> float:
    """공백 제외 문자 중 한글 음절 비율."""
    stripped = re.sub(r"\s", "", text)
    if not stripped:
        return 0.0
    return len(_hangul.findall(stripped)) / len(stripped)


def repetition_ratio(text: str, n: int = 20) -> float:
    """문자 n-gram 중복 비율 — 보일러플레이트·스팸 감지. 블록이 k번 반복되면 약 1 - 1/k.

    모든 위치에서 n-gram을 뗀다(슬라이딩). 예전에는 0, 20, 40 … 자리에서만 떼서, 길이가 20의
    배수와 어긋나는 문단(예: 507자)을 10번 반복해도 n-gram이 전부 달라 0.0이 나왔다.
    """
    if len(text) < n * 2:
        return 0.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


VAL_FRACTION = 0.005  # data.prepare와 data.prepare_anneal이 같은 값을 써야 한다


def is_val(text: str, fraction: float = VAL_FRACTION) -> bool:
    """문서 내용의 해시로 val 문서인지 정한다.

    예전에는 '처음 나온 문서 N토큰'을 val로 썼다. 그러면 (1) annealing이 교과서·위키·
    instruction을 dedup 없이 여러 번 반복할 때 val 문서가 2회차부터 train에 들어갔고,
    (2) annealing이 같은 데이터셋을 처음부터 다시 읽어 사전학습 val 문서를 train에 넣었다.
    해시로 정하면 같은 문서는 반복 회차·스크립트·순서와 무관하게 늘 같은 쪽이다.
    """
    h = hashlib.blake2b(text[:2000].encode("utf-8", "surrogatepass"), digest_size=8,
                        person=b"bebut-val").digest()
    return int.from_bytes(h, "big") < fraction * 2**64


def keep_document(text: str, korean_source: bool = True) -> bool:
    if len(text) < 200:  # 너무 짧은 문서는 문맥 학습에 도움 안 됨
        return False
    if korean_source and hangul_ratio(text) < 0.3:
        return False
    if repetition_ratio(text) > 0.5:
        return False
    return True


class MinHashDeduper:
    """MinHash 서명 기반 근사 중복 제거.

    문서를 문자 5-gram 집합으로 보고, num_perm개의 해시 함수 각각에 대한 최소값을
    서명으로 삼는다. 두 문서의 서명이 밴드 단위로 하나라도 일치하면 중복으로 판정
    (LSH banding). 수백만 문서까지 메모리 내 처리 가능.
    """

    _MERSENNE = (1 << 61) - 1

    def __init__(self, num_perm: int = 64, bands: int = 8, ngram: int = 5, seed: int = 42):
        assert num_perm % bands == 0
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.ngram = ngram
        self.seen: list[set[bytes]] = [set() for _ in range(bands)]
        # gram당 해시는 1번만 하고, num_perm개의 순열은 (a*h+b) mod p 로 근사
        import random

        rng = random.Random(seed)
        self.perms = [
            (rng.randrange(1, self._MERSENNE), rng.randrange(self._MERSENNE))
            for _ in range(num_perm)
        ]

    def _signature(self, text: str) -> list[int]:
        grams = {text[i : i + self.ngram] for i in range(len(text) - self.ngram + 1)}
        if not grams:
            return [0] * self.num_perm
        hashes = [
            int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
            for g in grams
        ]
        return [min((a * h + b) % self._MERSENNE for h in hashes) for a, b in self.perms]

    def band_keys(self, text: str) -> list[bytes]:
        """밴드별 해시 키 — 계산 비용의 대부분이 여기라 워커에서 병렬 실행 가능."""
        sig = self._signature(text)
        return [
            hashlib.blake2b(str(sig[b * self.rows : (b + 1) * self.rows]).encode(),
                            digest_size=8).digest()
            for b in range(self.bands)
        ]

    def check_keys(self, keys: list[bytes]) -> bool:
        """미리 계산된 키로 중복 판정 + 등록. 집합 조회뿐이라 매우 빠르다."""
        dup = any(k in self.seen[b] for b, k in enumerate(keys))
        if not dup:
            for b, k in enumerate(keys):
                self.seen[b].add(k)
        return dup

    def is_duplicate(self, text: str) -> bool:
        """중복이면 True. 아니면 등록하고 False."""
        return self.check_keys(self.band_keys(text))
