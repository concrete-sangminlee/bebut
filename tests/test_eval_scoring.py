"""KOBEST 채점 회귀 테스트.

예전 구현은 문맥 끝 공백이 선택지 첫 단어에 붙어 토큰화되는 것을 몰라
선택지 첫 토큰을 채점에서 빠뜨렸다 (BoolQ '예'는 채점 토큰 0개).
"""

import torch
import torch.nn.functional as F

from eval import (FORMATTERS, choice_logprob, choice_span, evaluate_choices,
                  normalized_score, score_continuations)
from tests.conftest import make_model

CASES = [
    ("문단...\n질문: 이것은 사실인가?\n답: ", "예"),
    ("문단...\n질문: 이것은 사실인가?\n답: ", "아니오"),
    ("비가 왔다. 그래서 ", "땅이 젖었다."),
    ("남자가 공을 던진다. ", "개가 공을 쫓아간다."),
    ("답: ", "예"),
    ("Context without trailing space", " and a choice"),
    ("숫자 문제:", "12345"),
    ("줄바꿈 문맥\n", "다음 줄"),
]


def test_choice_tokens_cover_whole_choice(real_tok):
    for ctx, ch in CASES:
        ids, start = choice_span(real_tok, ctx, ch)
        scored = real_tok.decode(ids[start:])
        prefix = real_tok.decode(ids[:start])
        # 채점 토큰은 선택지 전체(+ 문맥에서 넘어온 공백)를 덮고, 문맥은 그 앞에서 끝난다
        assert scored.endswith(ch), (ctx, ch, scored)
        assert prefix + scored == ctx + ch
        assert ctx.startswith(prefix)


def test_boolq_regression_counts(real_tok):
    """버그 상태에선 '예' 0토큰, '아니오' 1토큰이 채점됐다."""
    ctx = "문단...\n질문: 이것은 사실인가?\n답: "
    model = make_model(vocab_size=real_tok.vocab_size, dim=32, n_heads=2, n_kv_heads=1)
    _, n_yes = choice_logprob(model, real_tok, ctx, "예", "cpu")
    _, n_no = choice_logprob(model, real_tok, ctx, "아니오", "cpu")
    assert n_yes == len(real_tok.encode(" 예")) >= 1
    assert n_no == len(real_tok.encode(" 아니오")) >= 2


def brute_force_logprob(model, tok, ctx, ch):
    ids, start = choice_span(tok, ctx, ch)
    x = torch.tensor([ids[:-1]])
    logits, _ = model(x, targets=torch.tensor([ids[1:]]))
    logp = F.log_softmax(logits[0].float(), -1)
    return sum(logp[t - 1, ids[t]].item() for t in range(start, len(ids)))


def test_batched_scoring_matches_brute_force(tiny_tok):
    model = make_model(vocab_size=1024)
    pairs = [("비가 왔다. 그래서 ", "땅이 젖었다."), ("답: ", "예"),
             ("질문: 이것은 사실인가?\n답: ", "아니오"), ("The quick ", "brown fox")]
    batched = score_continuations(model, tiny_tok, pairs, "cpu")
    for (ctx, ch), s in zip(pairs, batched):
        assert abs(s - brute_force_logprob(model, tiny_tok, ctx, ch)) < 1e-4


def test_normalized_score_is_cond_minus_uncond(tiny_tok):
    model = make_model(vocab_size=1024)
    ctx, ch = "비가 왔다. 그래서 ", "땅이 젖었다."
    expect = brute_force_logprob(model, tiny_tok, ctx, ch) - brute_force_logprob(
        model, tiny_tok, "답: ", ch)
    assert abs(normalized_score(model, tiny_tok, ctx, ch, "cpu") - expect) < 1e-4


def test_evaluate_choices_end_to_end(tiny_tok):
    model = make_model(vocab_size=1024)
    rows = [{"premise": "비가 왔다.", "question": "결과", "alternative_1": "땅이 젖었다.",
             "alternative_2": "학교에 갑니다.", "label": i % 2} for i in range(6)]
    r = evaluate_choices(model, tiny_tok, FORMATTERS["copa"], rows, rows, 2, "cpu")
    assert r["n"] == 6
    for name in ("acc", "acc_len", "acc_calib"):
        assert 0.0 <= r[name] <= 1.0
        assert sum(r["preds"][name].values()) == 6
    # 선택지가 같으면 예측도 문항과 무관하게 같다 → 라벨이 반반이면 정확히 50%
    assert r["acc"] == 0.5
