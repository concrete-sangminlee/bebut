"""데이터 준비(data.prepare · data.prepare_anneal) 오프라인 e2e — 가짜 데이터셋, 실제 토크나이저.

예전에는 테스트가 없었다. 실제 실행에서 val 문서가 annealing train에 다시 들어갔고(교과서·위키·
instruction 반복, 같은 데이터셋 재독), 바닥난 소스가 아무 말 없이 빠졌고(위키 260M / 2.6B),
나무위키 2회 반복이 dedup으로 전부 버려졌다. 그 셋을 여기서 고정한다.
"""

import json
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest

import data.prepare as P
import data.prepare_anneal as A
from data.clean import is_val, repetition_ratio
from tokenizer.bpe import ByteBPETokenizer

TOK = "tokenizer/tokenizer.json"


def korean_doc(tag, n=260, seed=0):
    """keep_document를 통과하는(200자 이상, 한글 위주, 반복 없음) 문서. 앞에 고유 표지."""
    rng = random.Random(f"{tag}-{seed}")
    body = "".join(chr(0xAC00 + rng.randrange(11172)) + (" " if rng.random() < 0.2 else "")
                   for _ in range(n))
    return f"문서표지{tag} {body}"


class FakeStream:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


def read_docs(out_dir, split, tok):
    """shard를 읽어 문서 표지(문서표지XXX) 목록으로."""
    eos = tok.special_tokens["<|eos|>"]
    ids = np.concatenate([np.fromfile(os.path.join(out_dir, f), dtype=np.uint16)
                          for f in sorted(os.listdir(out_dir)) if f.startswith(split + "_")] or
                         [np.array([], dtype=np.uint16)])
    docs, cur = [], []
    for t in ids.tolist():
        if t == eos:
            docs.append(tok.decode(cur))
            cur = []
        else:
            cur.append(t)
    return [d.split()[0].removeprefix("문서표지") if "문서표지" in d.split()[0] else d for d in docs]


@pytest.fixture
def tok():
    return ByteBPETokenizer.load(TOK)


def test_repetition_ratio_catches_any_period():
    """예전엔 20자 격자에서만 n-gram을 떼서, 507자 문단 10번 반복이 0.0이었다."""
    rng = random.Random(0)
    for period in (500, 507, 333):
        block = "".join(chr(0xAC00 + rng.randrange(11172)) for _ in range(period))
        assert repetition_ratio(block * 10) > 0.85
    assert repetition_ratio(korean_doc("x", 3000)) < 0.05


def test_is_val_is_deterministic_and_proportional():
    docs = [korean_doc(i) for i in range(4000)]
    picked = [is_val(d, 0.1) for d in docs]
    assert picked == [is_val(d, 0.1) for d in docs]
    assert 0.07 < sum(picked) / len(picked) < 0.13


def test_prepare_hash_val_reports_exhausted_source_and_refuses_old_shards(tmp_path, monkeypatch, tok, capsys):
    data = {"web": [{"text": korean_doc(f"w{i}")} for i in range(300)],
            "wiki": [{"text": korean_doc(f"k{i}")} for i in range(5)]}  # 위키는 금방 바닥난다
    monkeypatch.setattr(P, "MIX", [("web", None, 0.8, True), ("wiki", None, 0.2, True)])
    monkeypatch.setattr(P, "load_dataset", lambda name, config, split, streaming: FakeStream(data[name]))
    monkeypatch.setattr(P, "SHARD_TOKENS", 5_000)  # shard가 여러 개로 넘어가는 경로도 지나게
    out = str(tmp_path / "shards")
    args = SimpleNamespace(tokens=120_000, tokenizer=TOK, out_dir=out, val_ratio=0.2, workers=2, overwrite=False)
    meta = P.run(args)
    assert "[wiki/None] 데이터 소진" in capsys.readouterr().out
    src = {s["name"]: s for s in meta["sources"]}
    assert src["wiki/None"]["exhausted"] and src["wiki/None"]["docs"] == 5
    assert meta["realized_mix"]["wiki/None"] < 0.1  # 설정 20%가 아니라 실제 비율이 남는다
    val, train = read_docs(out, "val", tok), read_docs(out, "train", tok)
    texts = {f"w{i}": korean_doc(f"w{i}") for i in range(300)} | {f"k{i}": korean_doc(f"k{i}") for i in range(5)}
    assert val and not set(val) & set(train)
    assert all(is_val(texts[d], 0.2) for d in val) and not any(is_val(texts[d], 0.2) for d in train)
    with pytest.raises(SystemExit):  # 예전 shard가 남아 학습에 섞이지 않게
        P.run(args)
    args.overwrite = True
    P.run(args)
    again = read_docs(out, "val", tok)
    # 예산 도달은 imap이 미리 읽은 만큼 늦게 판정돼 끝의 몇 문서는 실행마다 다를 수 있다
    n = min(len(again), len(val))
    assert again[:n] == val[:n] and n >= len(val) - 10


def test_anneal_val_never_reaches_train_and_dedup_repeat_counts(tmp_path, monkeypatch, tok):
    books = [{"text": korean_doc(f"b{i}")} for i in range(40)]
    namu = [{"text": korean_doc(f"n{i}")} for i in range(40)]
    inst = [{"instruction": f"질문{i} " + korean_doc(f"q{i}", 60), "output": f"답{i}"} for i in range(40)]
    rows = {"books": books, "namu": namu, "inst": inst}
    monkeypatch.setattr(A, "SOURCES", [
        dict(label="교과서", parts=[("books", None)], ratio=0.4, threshold=None, repeat=3, korean=True, dedup=False),
        dict(label="나무위키", parts=[("namu", None)], ratio=0.4, threshold=None, repeat=2, korean=True, dedup=True),
        dict(label="instruction", parts=[("inst", None)], ratio=0.2, threshold=None, repeat=3, korean=True,
             dedup=False, instruction=True),
    ])
    monkeypatch.setattr(P, "load_dataset", lambda name, config, split, streaming: FakeStream(rows[name]))
    out = str(tmp_path / "anneal")
    meta = A.run(SimpleNamespace(tokens=10_000_000, tokenizer=TOK, quality="data/quality_ko.npz", out_dir=out,
                                 val_ratio=0.25, workers=2, overwrite=False))
    val, train = read_docs(out, "val", tok), read_docs(out, "train", tok)
    assert val and not set(val) & set(train)  # 예전엔 교과서·instruction val 문서가 2회차부터 train에
    src = {s["label"]: s for s in meta["sources"]}
    n_namu_train = sum(1 for d in train if d.startswith("n"))
    n_namu_val = sum(1 for d in val if d.startswith("n"))
    assert n_namu_train == 2 * (40 - n_namu_val)  # 2회 반복이 실제로 2회분 (예전엔 1회분)
    assert src["교과서"]["val_skipped"] == 2 * sum(1 for d in val if d.startswith("b"))
    assert all(s["exhausted"] for s in meta["sources"])  # 예산을 못 채웠다
    saved = json.load(open(os.path.join(out, "meta.json"), encoding="utf-8"))
    assert saved["val_split"] == "hash" and set(saved["realized_mix"]) == {"교과서", "나무위키", "instruction"}


def test_nfd_document_is_kept_and_encoded_like_nfc(tok):
    import unicodedata

    P._init_worker(TOK)
    doc = korean_doc("nfd", 400)
    keys, ids, _ = P._process_doc((unicodedata.normalize("NFD", doc), True, 0.0))
    keys2, ids2, _ = P._process_doc((doc, True, 0.0))
    assert ids is not None and ids == ids2 and keys == keys2  # 예전엔 한글 비율 0으로 버려졌다
