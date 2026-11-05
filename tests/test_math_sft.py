"""산술 SFT 데이터 검증: 조사, 풀이의 모든 등식, 학습/평가 겹침 없음."""

import re

from data.make_math_sft import generate, heldout_pairs, josa


def test_josa_follows_sino_korean_reading():
    cases = {(7, "은/는"): "은", (4, "으로/로"): "로", (100, "을/를"): "을", (8, "으로/로"): "로",
             (12, "과/와"): "와", (10, "과/와"): "과", (10, "으로/로"): "으로", (3, "으로/로"): "으로",
             (6, "으로/로"): "으로", (1, "으로/로"): "로", (0, "으로/로"): "으로", (25, "은/는"): "는",
             (1000, "이/가"): "이", (20000, "을/를"): "을", (19, "을/를"): "를"}
    for (n, pair), want in cases.items():
        assert josa(n, pair) == want, (n, pair)


def _check_equalities(text):
    """풀이 줄의 'x = y = z'가 전부 같은 값인지 — 식을 직접 계산해 본다."""
    line = text.splitlines()[0] if "\n" in text else text
    for eq in re.findall(r"[\d(][\d\s+\-×÷()=]*=[\d\s+\-×÷()=]*\d", line):
        parts = [p.strip() for p in eq.split("=")]
        vals = {eval(p.replace("×", "*").replace("÷", "//")) for p in parts if p}  # noqa: S307
        assert len(vals) == 1, (eq, vals)


def test_every_solution_is_arithmetically_correct():
    train, test = generate(3000, 100, seed=1)
    ops = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
           "mul": lambda a, b: a * b, "div": lambda a, b: a // b}
    for row in train:
        m = row["meta"]
        res = ops[m["op"]](m["a"], m["b"])
        ans = row["messages"][1]["content"]
        assert str(res) in ans
        if not m["word"]:
            assert ans.endswith(f"따라서 답은 {res}입니다.")
        _check_equalities(ans)
        if m["op"] == "div":
            assert m["a"] % m["b"] == 0
    for item in test:
        m = item["meta"]
        assert item["a"] == [str(ops[m["op"]](m["a"], m["b"]))]


def test_no_overlap_between_train_and_test_and_original_heldout():
    train, test = generate(5000, 200, seed=2)
    key = lambda r: (r["meta"]["op"], r["meta"]["a"], r["meta"]["b"])  # noqa: E731
    train_keys = {key(r) for r in train}
    test_keys = {key(r) for r in test}
    assert not train_keys & test_keys
    banned = heldout_pairs()
    assert len(banned) == 5  # 원래 held-out의 계산 문항 5개가 모두 인식된다
    assert not train_keys & banned and not test_keys & banned
    assert len(test) == 200 and len(test_keys) == 200


def test_deterministic():
    assert generate(200, 20, seed=5) == generate(200, 20, seed=5)
    assert generate(200, 20, seed=5) != generate(200, 20, seed=6)
