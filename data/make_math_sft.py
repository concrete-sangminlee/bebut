"""계산 약점 보완용 합성 SFT 데이터 + 겹치지 않는 평가셋.

SFT 평가에서 계산은 0%였다 ("6 곱하기 7은 28입니다"). 사전학습 데이터엔 정답이 붙은
산수 문제가 드물고, SFT 데이터(KoAlpaca·KULLM)에도 거의 없다. 1B급 모델도 자릿수가
작은 사칙연산은 충분한 예제로 배울 수 있으므로, 풀이 과정을 짧게 보여주는 대화를
만들어 SFT에 섞는다 (python sft.py ... --extra-data data/math_sft.jsonl).

풀이는 '자리별로 쪼개서' 보여준다 — 47 + 38 = (40 + 30) + (7 + 8) = 70 + 15 = 85.
작은 모델은 답만 외우는 것보다 중간 단계를 거칠 때 훨씬 잘 맞힌다(chain-of-thought).

**평가 오염 방지가 핵심이다.** (연산, a, b) 조합을 해시로 학습/평가 칸에 나눠 같은 문제가
양쪽에 나오지 않게 하고, 기존 held-out 문항(evals/heldout_ko.json의 '계산')의 조합은
학습에서 뺀다. 문장 표현만 바꾼 같은 문제도 오염이므로 숫자 조합 단위로 나누고, 덧셈·곱셈은
순서를 바꾼 문제(9 × 3 / 3 × 9)도 같은 칸에 넣는다. 평가 문항은 마지막 숫자로 채점한다.

한국어 조사(은/는, 을/를, 와/과, 으로/로)는 숫자를 한자어로 읽었을 때 마지막 음절의
받침으로 정한다: 7(칠)은, 4(사)로, 100(백)을, 8(팔)로 — ㄹ 받침은 '으로'가 아니라 '로'.

사용:
  python -m data.make_math_sft --n 20000 --out data/math_sft.jsonl \\
      --test-out evals/math_heldout_ko.json --test-n 200
  python sft.py --base checkpoints/base_1b/latest.pt --config sft_1b --extra-data data/math_sft.jsonl
  python sft_eval.py --questions evals/math_heldout_ko.json --skip-kobest --models 최종=...
"""

import argparse
import hashlib
import json
import os
import random
import re

HELDOUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "evals", "heldout_ko.json")

# 한자어 수사의 마지막 음절과 받침 (None = 받침 없음)
_DIGIT_SYLLABLE = {0: "영", 1: "일", 2: "이", 3: "삼", 4: "사", 5: "오", 6: "육", 7: "칠", 8: "팔", 9: "구"}
_BATCHIM = {"영": "ㅇ", "일": "ㄹ", "이": None, "삼": "ㅁ", "사": None, "오": None, "육": "ㄱ",
            "칠": "ㄹ", "팔": "ㄹ", "구": None, "십": "ㅂ", "백": "ㄱ", "천": "ㄴ", "만": "ㄴ"}


def last_syllable(n: int) -> str:
    """숫자를 한자어로 읽을 때의 마지막 음절. 12 → 이, 20 → 십, 100 → 백, 0 → 영."""
    n = abs(n)
    if n == 0:
        return "영"
    for unit, name in ((10, None), (100, "십"), (1000, "백"), (10000, "천")):
        if n % unit:
            return _DIGIT_SYLLABLE[n % 10] if name is None else name
    return "만"


def josa(n: int, pair: str) -> str:
    """숫자 뒤 조사. pair는 '은/는', '이/가', '을/를', '과/와', '으로/로' 중 하나."""
    with_b, without_b = pair.split("/")
    b = _BATCHIM[last_syllable(n)]
    if pair == "으로/로":
        return "로" if b in (None, "ㄹ") else "으로"
    return with_b if b else without_b


def J(n: int, pair: str) -> str:
    return f"{n}{josa(n, pair)}"


# ---------------- 문제 · 풀이 ----------------

def _split_tens(n: int) -> tuple[int, int]:
    return n - n % 10, n % 10


def solve_add(a, b):
    if a < 10 or b < 10:
        return f"{a} + {b} = {a + b}"
    (at, ao), (bt, bo) = _split_tens(a), _split_tens(b)
    return f"{a} + {b} = ({at} + {bt}) + ({ao} + {bo}) = {at + bt} + {ao + bo} = {a + b}"


def solve_sub(a, b):
    if b < 10:
        return f"{a} - {b} = {a - b}"
    bt, bo = _split_tens(b)
    return f"{a} - {b} = {a} - {bt} - {bo} = {a - bt} - {bo} = {a - b}"


def solve_mul(a, b):
    return f"{a} × {b} = {a * b}"


def solve_mul2(a, b):  # a는 두 자리 이상, b는 한 자리
    at, ao = _split_tens(a)
    if ao == 0:
        return f"{a} × {b} = {a * b}"
    return f"{a} × {b} = {at} × {b} + {ao} × {b} = {at * b} + {ao * b} = {a * b}"


def solve_div(a, b):
    return f"{a} ÷ {b} = {a // b} (확인: {b} × {a // b} = {a})"


TEMPLATES = {
    "add": [
        lambda a, b: f"{a} 더하기 {J(b, '은/는')} 얼마야?",
        lambda a, b: f"{J(a, '과/와')} {J(b, '을/를')} 더하면?",
        lambda a, b: f"{a} + {J(b, '은/는')} 얼마인가요?",
        lambda a, b: f"{a}에 {J(b, '을/를')} 더하면 얼마야?",
        lambda a, b: f"{J(a, '과/와')} {b}의 합을 구해줘.",
        lambda a, b: f"{a} + {b} = ?",
    ],
    "sub": [
        lambda a, b: f"{a}에서 {J(b, '을/를')} 빼면 얼마야?",
        lambda a, b: f"{a} 빼기 {J(b, '은/는')}?",
        lambda a, b: f"{a} - {J(b, '은/는')} 얼마인가요?",
        lambda a, b: f"{J(a, '과/와')} {b}의 차를 구해줘.",
        lambda a, b: f"{a} - {b} = ?",
    ],
    "mul": [
        lambda a, b: f"{a} 곱하기 {J(b, '은/는')} 얼마야?",
        lambda a, b: f"{J(a, '과/와')} {J(b, '을/를')} 곱하면?",
        lambda a, b: f"{a} × {J(b, '은/는')} 얼마인가요?",
        lambda a, b: f"{a} * {b} = ?",
    ],
    "div": [
        lambda a, b: f"{J(a, '을/를')} {J(b, '으로/로')} 나누면 얼마야?",
        lambda a, b: f"{a} 나누기 {J(b, '은/는')}?",
        lambda a, b: f"{a} ÷ {J(b, '은/는')} 얼마인가요?",
        lambda a, b: f"{a} / {b} = ?",
    ],
}

OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
       "mul": lambda a, b: a * b, "div": lambda a, b: a // b}

WORD = [  # (질문, 풀이 문장) — 남은 개수를 묻는 뺄셈, 모두 합치는 덧셈, 묶음 곱셈
    ("sub", lambda a, b: f"사과가 {a}개 있는데 {b}개를 먹었어. 몇 개 남았어?",
     lambda a, b: f"{a}개에서 {b}개를 먹었으니 {solve_sub(a, b)}, {a - b}개가 남았습니다."),
    ("add", lambda a, b: f"연필이 {a}자루 있고 친구에게 {b}자루를 더 받았어. 모두 몇 자루야?",
     lambda a, b: f"{a}자루에 {b}자루를 더하면 {solve_add(a, b)}, 모두 {a + b}자루입니다."),
    ("mul", lambda a, b: f"한 상자에 귤이 {a}개씩 들어 있어. {b}상자면 귤은 모두 몇 개야?",
     lambda a, b: f"{a}개씩 {b}상자이므로 {solve_mul2(a, b) if a >= 10 else solve_mul(a, b)}, 모두 {a * b}개입니다."),
    ("div", lambda a, b: f"쿠키 {a}개를 {b}명이 똑같이 나눠 가지면 한 사람당 몇 개야?",
     lambda a, b: f"{a}개를 {b}명에게 똑같이 나누면 {solve_div(a, b)}, 한 사람당 {a // b}개입니다."),
]


def sample_operands(op, rng):
    """자릿수를 고루 섞되 작은 수를 더 자주 — 모델이 확실히 맞혀야 할 범위부터."""
    if op == "add":
        hi = rng.choice([10, 100, 100, 1000])
        return rng.randint(0, hi), rng.randint(0, hi)
    if op == "sub":
        a = rng.randint(1, rng.choice([20, 100, 100, 1000]))
        return a, rng.randint(0, a)
    if op == "mul":
        if rng.random() < 0.6:
            return rng.randint(1, 12), rng.randint(1, 12)  # 구구단 + 11·12단
        return rng.randint(10, 99), rng.randint(2, 9)
    b = rng.randint(2, 12)  # div: 나누어떨어지는 것만
    return b * rng.randint(1, 30), b


def canon(op, a, b):
    """덧셈·곱셈은 교환법칙이 있어 9 × 3과 3 × 9는 같은 문제다 — 같은 열쇠로 묶는다.

    순서 그대로 나누면 평가의 9 × 3이 학습에 3 × 9로 들어가 평가가 오염된다
    (기본 시드에서 구구단 평가 문항 14개 중 11개가 그랬다).
    """
    return (op, min(a, b), max(a, b)) if op in ("add", "mul") else (op, a, b)


def is_test_pair(op, a, b, test_ratio) -> bool:
    """(연산, a, b) → 평가 칸 여부. 해시라서 생성 순서·시드와 무관하게 늘 같은 칸이다."""
    op, a, b = canon(op, a, b)
    h = hashlib.blake2b(f"{op}:{a}:{b}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % 10_000 < test_ratio * 10_000


def heldout_pairs(path=HELDOUT):
    """기존 대화 평가의 계산 문항이 쓰는 (연산, a, b) — 학습에서 반드시 뺀다.

    경로는 이 파일 기준이다. 실행 위치에 따라 파일을 못 찾아 제외 목록이 조용히 비면
    오염된 데이터가 만들어지므로, 없으면 실패한다.
    """
    pats = [("add", r"(\d+)\s*더하기\s*(\d+)"), ("sub", r"(\d+)에서\s*(\d+)\S*\s*빼"),
            ("mul", r"(\d+)\s*곱하기\s*(\d+)"), ("div", r"(\d+)\S*\s*(\d+)\S*\s*나누"),
            ("sub", r"(\d+)개 있는데\s*(\d+)개를 먹")]
    out = set()
    with open(path, encoding="utf-8") as f:
        qs = json.load(f)
    for q in qs:
        if q.get("type") != "계산":
            continue
        for op, pat in pats:
            m = re.search(pat, q["q"])
            if m:
                out.add(canon(op, int(m.group(1)), int(m.group(2))))
    return out


def make_item(op, a, b, rng, word=False):
    """(질문, 풀이 답, 정답 숫자)."""
    if word:
        _, q, ans = rng.choice([w for w in WORD if w[0] == op])
        res = OPS[op](a, b)
        return q(a, b), ans(a, b), res
    q = rng.choice(TEMPLATES[op])(a, b)
    if op == "add":
        work, res = solve_add(a, b), a + b
    elif op == "sub":
        work, res = solve_sub(a, b), a - b
    elif op == "mul":
        work = solve_mul2(a, b) if a >= 10 and b < 10 else solve_mul(a, b)
        res = a * b
    else:
        work, res = solve_div(a, b), a // b
    return q, f"{work}\n따라서 답은 {res}입니다.", res


def generate(n, test_n, seed=0, test_ratio=0.1, word_ratio=0.25):
    """(학습 대화 목록, 평가 문항 목록). 둘의 (연산, a, b)는 겹치지 않는다."""
    rng = random.Random(seed)
    banned = heldout_pairs()
    train, test, seen_test = [], [], set()
    ops = ["add", "sub", "mul", "div"]
    guard = 0
    while (len(train) < n or len(test) < test_n) and guard < 50 * (n + test_n):
        guard += 1
        op = rng.choice(ops)
        a, b = sample_operands(op, rng)
        word = rng.random() < word_ratio and (op != "mul" or b < 10)
        key = canon(op, a, b)
        if is_test_pair(op, a, b, test_ratio) or key in banned:
            res = OPS[op](a, b)
            # 답이 피연산자와 같으면(11 × 1, 15 − 0) 질문을 따라 말하기만 해도 맞는다 — 평가에서 뺀다
            trivial = res in (a, b)
            if len(test) < test_n and key not in seen_test and key not in banned and not trivial:
                seen_test.add(key)
                q, _, res = make_item(op, a, b, rng, word)
                # 풀이 중간의 숫자가 우연히 정답과 같을 수 있어 '마지막 숫자'로 채점한다
                test.append({"type": "계산", "q": q, "a": [str(res)], "match": "last_number",
                             "meta": {"op": op, "a": a, "b": b, "word": word}})
            continue
        if len(train) < n:
            q, ans, _ = make_item(op, a, b, rng, word)
            # meta는 sft.py가 무시한다 — 겹침 검사·연산별 분석용
            train.append({"messages": [{"role": "user", "content": q},
                                       {"role": "assistant", "content": ans}],
                          "meta": {"op": op, "a": a, "b": b, "word": word}})
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="학습 대화 수")
    ap.add_argument("--out", default="data/math_sft.jsonl")
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--test-out", default="evals/math_heldout_ko.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, test = generate(args.n, args.test_n, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.test_out, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=1)
    print(f"학습 {len(train)}개 → {args.out} | 평가 {len(test)}개 → {args.test_out}")
    print("예시:", json.dumps(train[0]["messages"], ensure_ascii=False))


if __name__ == "__main__":
    main()
