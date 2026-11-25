"""원문 기준 bits-per-byte (evals/text_bpb.py): 모든 토큰을 한 번씩, 창 설정과 무관하게."""

import math

import pytest
import torch
import torch.nn.functional as F

from evals.text_bpb import BebutScorer, lossless_docs, score_doc, score_text
from tests.conftest import make_model

DOCS = ["대한민국은 동아시아의 한반도 남부에 있는 나라이다. 수도는 서울이다.",
        "The quick brown fox jumps over the lazy dog. 12,345원입니다.",
        "짧은 글"]


class BigramScorer:
    """logits가 바로 앞 토큰에만 달린 가짜 모델 — 문맥을 어떻게 자르든 결과가 같아야 한다."""

    name, max_len, params = "bigram", 1 << 30, 0

    def __init__(self, vocab=50, seed=0):
        self.table = torch.randn(vocab, vocab, generator=torch.Generator().manual_seed(seed))
        self.start_id, self.calls = 0, []

    def encode(self, text):
        return [1 + (b % 49) for b in text.encode("utf-8")]

    def lossless(self, text, ids):
        return "깨짐" not in text

    def logits(self, ids):
        self.calls.append(len(ids))
        return self.table[torch.tensor(ids)]


def brute_force(scorer, ids):
    seq = [scorer.start_id] + ids
    lp = F.log_softmax(scorer.table[torch.tensor(seq[:-1])], dim=-1)
    return -float(lp.gather(1, torch.tensor(seq[1:])[:, None]).sum())


def test_every_token_scored_once_whatever_the_window():
    s = BigramScorer()
    ids = s.encode("가나다라마바사아자차카타파하" * 9)  # 378토큰
    want = brute_force(s, ids)
    for max_len, stride in ((1000, 500), (64, 32), (64, 63), (64, 1), (17, 5), (2, 1)):
        nats, n = score_doc(s, ids, max_len, stride)
        assert n == len(ids) and nats == pytest.approx(want, rel=1e-5), (max_len, stride)
    s.calls.clear()
    score_doc(s, ids, 64, 32)
    assert max(s.calls) <= 64  # 창을 넘겨 모델에 넣지 않는다
    with pytest.raises(ValueError):
        score_text(s, DOCS, max_len=64, stride=64)


def test_window_sweep_is_exhaustive_for_short_documents():
    """문서 길이 1~40 · 창 2~12 · 가능한 모든 stride: 어떤 조합에서도 토큰을 빠뜨리거나 두 번 세지 않고
    (합이 통째로 잰 값과 같다), 창보다 긴 입력을 모델에 넣지 않는다."""
    s = BigramScorer()
    base = s.encode("가나다라마바사아자차카타파하")
    for n in range(1, 41):
        ids = base[:n]
        want = brute_force(s, ids)
        for max_len in range(2, 13):
            for stride in range(1, max_len):
                s.calls.clear()
                nats, scored = score_doc(s, ids, max_len, stride)
                assert scored == n and nats == pytest.approx(want, rel=1e-5, abs=1e-6), (n, max_len, stride)
                assert max(s.calls) <= max_len


def test_bpb_is_bits_over_utf8_bytes_and_lossy_docs_are_dropped():
    s = BigramScorer()
    r = score_text(s, DOCS, max_len=32, stride=16)
    bits = sum(brute_force(s, s.encode(d)) for d in DOCS) / math.log(2)
    nbytes = sum(len(d.encode("utf-8")) for d in DOCS)
    assert r["bytes"] == nbytes and r["bpb"] == pytest.approx(bits / nbytes, rel=1e-5)
    assert r["tokens"] == nbytes and r["bytes_per_token"] == 1.0  # 이 가짜 토크나이저는 바이트당 한 토큰
    keep, dropped = lossless_docs([s], DOCS + ["깨짐 문서"])
    assert keep == DOCS and dropped == {"bigram": 1}


def test_bebut_scorer_matches_direct_forward(tiny_tok):
    """한 창에 들어가는 문서는 모델을 한 번 돌린 cross-entropy 합과 같다. 긴 문서는 창을 밀며 잰다."""
    torch.manual_seed(0)
    model = make_model(vocab_size=1024, max_seq_len=64)
    s = BebutScorer(model, tiny_tok)
    ids = tiny_tok.encode(DOCS[0])
    assert 1 < len(ids) < 60
    seq = torch.tensor([[s.start_id] + ids])
    with torch.no_grad():
        logits = model.lm_head(model.hidden_states(seq[:, :-1]))[0]
    want = float(F.cross_entropy(logits.float(), seq[0, 1:], reduction="sum"))
    nats, n = score_doc(s, ids, 64, 32)
    assert n == len(ids) and nats == pytest.approx(want, rel=1e-5)
    long_ids = tiny_tok.encode(" ".join(DOCS) * 6)
    assert len(long_ids) > 64
    nats, n = score_doc(s, long_ids, 64, 32)  # 문맥(64)을 넘는 문서도 죽지 않고 전부 채점
    assert n == len(long_ids) and math.isfinite(nats)
    r = score_text(s, DOCS, max_len=1024, stride=512)  # max_len은 모델 문맥(64)으로 줄어든다
    assert r["docs"] == 3 and 0 < r["bpb"] < 40


def test_merge_baselines_only_joins_the_same_text(tmp_path):
    """--baseline: 같은 글 · 같은 창으로 잰 예전 결과만 한 표에 합친다 (글이 다르면 BPB는 비교 불가)."""
    import json

    from evals.text_bpb import merge_baselines

    def result(models, sha="aaaa", max_len=1024):
        return {"max_len": max_len, "stride": 512, "text": {"웹": {"sha256": sha}},
                "models": {m: {"params": 10, "domains": {"웹": {"bpb": b}}} for m, b in models.items()}}

    paths = []
    short = result({"짧은문맥": 0.3})  # 파일에는 요청한 창(1024)이 적혀 있지만 이 모델은 512로 쟀다
    short["models"]["짧은문맥"]["domains"]["웹"].update(max_len=512, stride=256)
    for k, r in enumerate([result({"공개A": 1.2, "공개B": 0.9}), result({"다른글": 0.1}, sha="bbbb"),
                           result({"다른창": 0.2}, max_len=512), result({"bebut": 9.9}), short]):
        paths.append(str(tmp_path / f"b{k}.json"))
        json.dump(r, open(paths[-1], "w", encoding="utf-8"))

    table, skipped = merge_baselines(result({"bebut": 1.0}), paths)
    assert [(m, b) for m, b, _, _ in table["웹"]] == [("공개B", 0.9), ("bebut", 1.0), ("공개A", 1.2)]  # BPB 순
    assert table["웹"][1][3] == "이번"  # 이름이 같으면 이번에 잰 값이 이긴다
    assert len(skipped) == 3 and "sha256" in skipped[0] and "문맥 창" in skipped[1] and "짧은문맥" in skipped[2]


def test_short_context_model_records_the_window_it_actually_used():
    """모델 문맥이 요청한 창보다 짧으면 줄여서 재고, 실제로 쓴 창을 결과에 남긴다."""
    scorer = BigramScorer()
    scorer.max_len = 8
    r = score_text(scorer, ["가나다라마바사아자차카타파하" * 3], max_len=1024, stride=512)
    assert (r["max_len"], r["stride"]) == (8, 4)
    assert score_text(BigramScorer(), ["가나다"], max_len=16, stride=8)["max_len"] == 16


def test_cli_stops_when_no_document_survives(monkeypatch, tmp_path, tiny_tok):
    """남은 문서가 없으면 멈춘다 — 예전엔 그대로 진행해 모든 모델이 BPB 0.0(가장 좋은 점수)으로 저장됐다."""
    import sys

    from evals import text_bpb

    tiny_tok.save(str(tmp_path / "tok.json"))
    monkeypatch.setattr(text_bpb, "load_docs", lambda kb, text_file=None: {"웹": []})
    monkeypatch.setattr(sys, "argv", ["text_bpb", "--ckpt", "x=missing.pt", "--tokenizer", str(tmp_path / "tok.json"),
                                      "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="하나도 없습니다"):
        text_bpb.main()
    assert not (tmp_path / "o.json").exists()
