"""평가 점수의 불확실성 — 신뢰구간과 두 모델 비교 검정.

이 프로젝트의 결론 여럿이 잡음 안에 있었다. 대화 정답률 '74.2% vs 71.0%'는 한 문제 차이였고,
마일스톤 KOBEST는 150문항(95% 구간 ±8%p)으로 재면서 0~3%p 변화를 '정체 뒤 상승'으로 읽었다.
점수 하나에는 Wilson 구간을, 두 모델의 차이에는 검정을 붙여 읽는다.

두 모델을 **같은 문항**으로 쟀다면 짝 비교가 훨씬 정밀하다. 문항 난이도의 변동이 상쇄되고,
둘이 다르게 답한 문항(불일치 쌍)만 정보가 되기 때문이다 — McNemar 검정. 문항별 정오가 없고
정답 수만 남은 예전 기록은 짝 없는 비교(두 비율의 차이)로 볼 수밖에 없다.
"""

import math

Z95 = 1.959964


def wilson_ci(k: int, n: int, z: float = Z95) -> list[float]:
    """정답 k / n의 95% Wilson 신뢰구간 [lo, hi]. 문항이 적을수록 넓다 — 대화 평가 30문항이면
    70%의 구간이 대략 52~83%라, 한두 문제 차이는 구분되지 않는다."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def half_width(n: int, p: float = 0.5) -> float:
    """n문항으로 잰 정확도의 95% 구간 반폭(정규 근사). p=0.5가 가장 넓다: n=150 → ±8%p, 500 → ±4.4%p."""
    return Z95 * math.sqrt(p * (1 - p) / n) if n else 1.0


def _two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def mcnemar(a_only: int, b_only: int) -> float:
    """짝 비교의 양측 p값. a_only: A만 맞힌 문항 수, b_only: B만 맞힌 문항 수.

    둘 다 맞히거나 둘 다 틀린 문항은 차이에 대해 아무것도 말해 주지 않는다. '두 모델이 같다'면
    불일치 쌍은 반반으로 갈려야 한다 — 그 이항 분포의 정확한 꼬리 확률. 쌍이 많으면(>2000)
    연속성 보정한 정규 근사를 쓴다.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    if n > 2000:
        return _two_sided((abs(a_only - b_only) - 1) / math.sqrt(n)) if a_only != b_only else 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_diff(a: str, b: str) -> dict:
    """문항별 정오 문자열('1' 맞힘 · '0' 틀림 · 그 밖은 채점 제외) 둘 → B − A의 차이와 검정.

    구간은 Bonett–Price(2012)의 보정 Wald 구간: 한쪽만 맞힌 수 두 칸에 1씩 더하고(n에는 2)
    정규 근사를 쓴다. 보정하지 않은 Wald 구간은 엇갈린 문항이 없으면 폭이 0이 되고(30문항에서 두
    모델이 똑같이 답했다고 차이가 정확히 0이라고 할 수는 없다), 작은 표본에서 실제 포함률이 95%에
    못 미친다 — tests/test_stats.py가 포함률을 직접 센다.
    """
    if len(a) != len(b):
        raise ValueError(f"문항 수가 다르다 ({len(a)} vs {len(b)}) — 같은 문항·같은 --limit으로 잰 결과여야 한다")
    pairs = [(x, y) for x, y in zip(a, b) if x in "01" and y in "01"]
    n = len(pairs)
    a_only = sum(x == "1" and y == "0" for x, y in pairs)
    b_only = sum(x == "0" and y == "1" for x, y in pairs)
    ka, kb = sum(x == "1" for x, _ in pairs), sum(y == "1" for _, y in pairs)
    return {"method": "paired", "n": n, "acc_a": ka / max(1, n), "acc_b": kb / max(1, n),
            "diff": (kb - ka) / max(1, n), "ci95": paired_ci(a_only, b_only, n), "p": mcnemar(a_only, b_only),
            "a_only": a_only, "b_only": b_only}


def paired_ci(a_only: int, b_only: int, n: int) -> list[float]:
    """짝 지은 n문항에서 B − A 정답률 차이의 95% 구간 (Bonett–Price 보정 Wald)."""
    pa, pb = (a_only + 1) / (n + 2), (b_only + 1) / (n + 2)
    mid = pb - pa
    half = Z95 * math.sqrt((pa + pb - mid * mid) / (n + 2))
    return [max(-1.0, mid - half), min(1.0, mid + half)]


def unpaired_diff(ka: int, na: int, kb: int, nb: int) -> dict:
    """정답 수만 있을 때: B − A의 차이, Newcombe 구간(두 Wilson 구간을 합성), 두 비율 z검정.

    같은 문항으로 쟀더라도 문항별 정오가 없으면 이렇게밖에 못 본다 — 짝 비교보다 구간이 넓다.
    """
    pa, pb = ka / max(1, na), kb / max(1, nb)
    if na == 0 or nb == 0:  # 한쪽에 문항이 없으면 비교할 것이 없다 — 구간은 전체, p는 1
        return {"method": "unpaired", "n": 0, "acc_a": pa, "acc_b": pb, "diff": 0.0, "ci95": [-1.0, 1.0], "p": 1.0}
    la, ua = wilson_ci(ka, na)
    lb, ub = wilson_ci(kb, nb)
    diff = pb - pa
    lo = diff - math.sqrt((pb - lb) ** 2 + (ua - pa) ** 2)
    hi = diff + math.sqrt((ub - pb) ** 2 + (pa - la) ** 2)
    pool = (ka + kb) / max(1, na + nb)
    se = math.sqrt(pool * (1 - pool) * (1 / max(1, na) + 1 / max(1, nb)))
    return {"method": "unpaired", "n": min(na, nb), "acc_a": pa, "acc_b": pb, "diff": diff,
            "ci95": [lo, hi], "p": _two_sided(diff / se) if se > 0 else 1.0}


# ---------------- 평가 결과 JSON 읽기 ----------------


def _items(r, metric):
    """벤치마크 결과 하나 → 문항별 정오 문자열 (없으면 None). 과목별 결과는 과목 이름순으로 잇는다."""
    if "items" in r:
        return r["items"].get(metric)
    if "by_config" in r and all("items" in c for c in r["by_config"].values()):
        return "".join(r["by_config"][k]["items"][metric] for k in sorted(r["by_config"]))
    return None


def _counts(r, metric):
    """(정답 수, 문항 수). 예전 기록(acc가 퍼센트, 지표 하나)도 읽는다."""
    if "acc_calib" in r:  # 지금 형식: 0~1 비율, 지표 셋
        return round(r[metric] * r["n"]), r["n"]
    return round(r["acc"] / 100 * r["n"]), r["n"]


def compare_benchmark(a: dict, b: dict, metric: str = "acc_calib") -> dict:
    """같은 벤치마크의 두 결과 → 비교. 문항별 정오가 둘 다 있으면 짝 비교, 아니면 짝 없는 비교."""
    ia, ib = _items(a, metric), _items(b, metric)
    if ia is not None and ib is not None:
        return paired_diff(ia, ib)
    return unpaired_diff(*_counts(a, metric), *_counts(b, metric))


def compare_chat(a: dict, b: dict) -> dict | None:
    """sft_eval의 대화 평가(answers) 두 개 → 짝 비교. 채점 제외·오염대조·무채점 문항은 뺀다."""
    if "answers" not in a or "answers" not in b:
        return None

    def marks(r):
        return "".join("-" if x.get("correct") is None or x.get("excluded") or x.get("type") == "오염대조"
                       else str(int(x["correct"])) for x in r["answers"])

    return paired_diff(marks(a), marks(b))


def compare_results(a: dict, b: dict, metric: str = "acc_calib") -> dict:
    """final_eval · sft_eval 결과의 한 모델분(dict) 둘 → {이름: 비교}. 양쪽에 다 있는 것만."""
    out = {}
    chat = compare_chat(a, b)
    if chat:
        out["대화"] = chat
    for group in ("kobest", "bench"):
        for name in a.get(group) or {}:
            if name in (b.get(group) or {}):
                out[name] = compare_benchmark(a[group][name], b[group][name], metric)
    return out


def format_comparison(name: str, c: dict) -> str:
    kind = (f"짝 비교 n={c['n']}, A만 맞힘 {c['a_only']} · B만 맞힘 {c['b_only']}" if c["method"] == "paired"
            else f"짝 없는 비교 n={c['n']} — 문항별 기록이 없어 구간이 넓다")
    verdict = "유의" if c["p"] < 0.05 else "잡음과 구분 안 됨"
    return (f"  {name:<10} {c['acc_a']*100:5.1f}% → {c['acc_b']*100:5.1f}%  차이 {c['diff']*100:+5.1f}%p  "
            f"95% 구간 [{c['ci95'][0]*100:+.1f}, {c['ci95'][1]*100:+.1f}]  p={c['p']:.3f} ({verdict})  [{kind}]")


def print_comparisons(results: dict, metric: str = "acc_calib") -> None:
    """한 결과 파일 안의 모델들을 차례로 비교해 출력한다 (이웃한 쌍 + 셋 이상이면 처음 ↔ 끝)."""
    labels = list(results)
    pairs = list(zip(labels, labels[1:]))
    if len(labels) > 2:
        pairs.append((labels[0], labels[-1]))
    for a, b in pairs:
        res = compare_results(results[a], results[b], metric)
        if res:
            print(f"\n비교: {a} → {b}")
            for name, c in res.items():
                print(format_comparison(name, c))
