"""KMMLU · HAE-RAE 채점 검증. 행은 실제 데이터셋 스키마 그대로 (datasets-server에서 확인).

datasets 라이브러리·네트워크 없이 돌도록 가짜 datasets 모듈을 끼운다.
실제 데이터로도 돌려보려면: BEBUT_NET_TESTS=1 python -m pytest tests/test_benchmarks.py
"""

import math
import os
import sys
import types

import pytest

import eval as ev
from tests.conftest import make_model

KMMLU_ROW = {"question": "포도당의 특성이 아닌 것은?", "answer": 4, "A": "환원당", "B": "알도오스",
             "C": "헥소오스", "D": "퓨라노오스", "Category": "Biology", "Human Accuracy": 0.5915}
HAERAE_5 = {"question": "청동기 시대에 커다란 돌을 쌓아 만든 족장의 무덤을 무엇이라고 하는가?",
            "a": "고분", "b": "고인돌", "c": "돌무덤", "d": "돌기와집", "e": "굴식돌방", "answer": "b", "split": "HI"}
HAERAE_4 = {"question": ("<빈칸>에 들어갈 가장 알맞은 것을 고르십시오.  "
                         "### 참고: 외국에 나가려면 <빈칸>이 반드시 있어야 해요."),
            "a": "도장", "b": "명함", "c": "여권", "d": "통장", "e": None, "answer": "c", "split": "RC"}


def test_kmmlu_formatter_cloze_and_mc():
    ctx, ch, label = ev.make_kmmlu_formatter("cloze")(KMMLU_ROW)
    assert ctx.endswith("\n답: ") and ch == ["환원당", "알도오스", "헥소오스", "퓨라노오스"] and label == 3
    ctx, ch, label = ev.make_kmmlu_formatter("mc")(KMMLU_ROW)
    assert "\nA. 환원당\n" in ctx and ctx.endswith("D. 퓨라노오스\n정답: ")
    assert ch == ["A", "B", "C", "D"] and label == 3
    with pytest.raises(ValueError):
        ev.make_kmmlu_formatter()({**KMMLU_ROW, "answer": 0})


def test_haerae_formatter_handles_four_and_five_options():
    _, ch, label = ev.make_haerae_formatter()(HAERAE_5)
    assert len(ch) == 5 and ch[label] == "고인돌"
    # 4지선다 범주에선 e가 이렇게 비어 있다 (실제 데이터의 독해 범주는 float NaN)
    for empty in (None, float("nan"), "None", ""):
        _, ch, label = ev.make_haerae_formatter("mc")({**HAERAE_4, "e": empty})
        assert ch == ["A", "B", "C", "D"] and label == 2
    with pytest.raises(ValueError):
        ev.make_haerae_formatter()({**HAERAE_4, "answer": "e"})


def test_aggregate_micro_and_macro():
    pa = {"acc": {0: 4, 1: 6}, "acc_len": {0: 10}, "acc_calib": {1: 10}}
    pb = {"acc": {0: 30}, "acc_len": {2: 30}, "acc_calib": {0: 15, 3: 15}}
    per = {"a": {"n": 10, "acc": 0.5, "acc_len": 0.5, "acc_calib": 0.6, "truncated": 1, "preds": pa},
           "b": {"n": 30, "acc": 0.1, "acc_len": 0.2, "acc_calib": 0.2, "truncated": 0, "preds": pb}}
    r = ev._aggregate(per)
    assert r["n"] == 40 and r["truncated"] == 1 and r["by_config"]["a"]["truncated"] == 1
    # 예측 분포 합계 (mc 형식의 문자 쏠림 감지용)
    assert r["preds"]["acc"] == {0: 34, 1: 6} and r["preds"]["acc_calib"] == {0: 15, 1: 10, 3: 15}
    assert math.isclose(r["acc_calib"], (6 + 6) / 40) and math.isclose(r["macro_acc_calib"], 0.4)


def test_long_context_is_truncated_from_the_left(tiny_tok):
    """문맥이 max_seq_len을 넘으면 앞을 잘라 채점한다 (예전엔 shape 오류로 죽었다)."""
    model = make_model(vocab_size=1024, max_seq_len=32)
    ctx = "안녕하세요. 오늘 날씨가 정말 좋네요. " * 20 + "그래서 "
    stats = {}
    full = ev.score_continuations(model, tiny_tok, [(ctx, "학교에 갑니다.")], "cpu", stats)
    assert stats["truncated"] == 1 and math.isfinite(full[0])
    ids, start = ev.choice_span(tiny_tok, ctx, "학교에 갑니다.")
    cut_ids, cut_start, cut = ev.fit_span(ids, start, 32)
    assert cut and len(cut_ids) == 33 and cut_ids[cut_start:] == ids[start:]  # 선택지는 그대로
    with pytest.raises(ValueError):  # 선택지만으로 문맥을 넘으면 자를 수 없다
        ev.fit_span(list(range(100)), 2, 32)


class FakeDataset(list):
    """datasets.Dataset에서 쓰는 만큼만: 정수 인덱싱, len, 순회, select."""

    def __getitem__(self, i):
        r = super().__getitem__(i)
        return FakeDataset(r) if isinstance(i, slice) else r

    def select(self, idx):
        return FakeDataset(self[i] for i in idx)


def install_fake_datasets(monkeypatch, configs):
    mod = types.ModuleType("datasets")
    mod.get_dataset_config_names = lambda name: list(configs[name])
    def load_dataset(name, config, split=None):
        d = configs[name][config]
        return d[split] if split else d
    mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_kmmlu_and_haerae_details_end_to_end(monkeypatch, tiny_tok):
    rows = FakeDataset([KMMLU_ROW] * 7)
    hrows = FakeDataset([HAERAE_5] * 4 + [HAERAE_4] * 4)
    install_fake_datasets(monkeypatch, {
        "HAERAE-HUB/KMMLU": {"Biology": {"dev": rows[:5], "test": rows}, "Law": {"dev": rows[:5], "test": rows[:3]}},
        "HAERAE-HUB/HAE_RAE_BENCH_1.0": {"History": {"test": hrows}},
    })
    model = make_model(vocab_size=1024, max_seq_len=512)
    k = ev.kmmlu_details(model, tiny_tok, 5, "cpu")
    assert k["n"] == 10 and set(k["by_config"]) == {"Biology", "Law"}
    assert ev.kmmlu_details(model, tiny_tok, 10, "cpu", limit=1)["shots"] == 5  # dev가 5문항뿐
    assert sum(k["preds"]["acc_calib"].values()) == 10
    assert ev.kmmlu_details(model, tiny_tok, 5, "cpu", limit=2)["n"] == 4
    h = ev.haerae_details(model, tiny_tok, 3, "cpu", fmt="mc")
    assert h["n"] == 5  # 앞 3문항은 few-shot 예시로 쓰고 채점에서 뺀다
    for r in (k, h):
        assert 0 <= r["acc_calib"] <= 1 and 0 <= r["macro_acc_calib"] <= 1


@pytest.mark.skipif(os.environ.get("BEBUT_NET_TESTS") != "1", reason="실제 데이터 테스트는 BEBUT_NET_TESTS=1")
def test_real_datasets_smoke(real_tok):
    model = make_model(vocab_size=real_tok.vocab_size, dim=32, n_heads=2, n_kv_heads=1, max_seq_len=2048)
    r = ev.kmmlu_details(model, real_tok, 5, "cpu", limit=2, subjects=["Biology", "Law"])
    assert r["n"] == 4
    r = ev.haerae_details(model, real_tok, 2, "cpu", limit=2)
    assert len(r["by_config"]) == 6
    for task in ("copa", "boolq", "hellaswag"):
        assert ev.kobest_details(model, real_tok, task, 2, "cpu", limit=2)["n"] == 2


def test_exclude_ids_skip_contaminated_items(monkeypatch, tiny_tok):
    """오염 문항 id를 주면 그 문항만 빼고 채점한다. HAE-RAE는 few-shot으로 쓴 앞부분만큼
    행 번호를 당겨야 한다 (id는 test 전체 기준)."""
    rows = FakeDataset([KMMLU_ROW] * 7)
    hrows = FakeDataset([HAERAE_5] * 8)
    install_fake_datasets(monkeypatch, {
        "HAERAE-HUB/KMMLU": {"Biology": {"dev": rows[:5], "test": rows}},
        "HAERAE-HUB/HAE_RAE_BENCH_1.0": {"History": {"test": hrows}},
    })
    model = make_model(vocab_size=1024, max_seq_len=512)
    k = ev.kmmlu_details(model, tiny_tok, 5, "cpu", exclude={"kmmlu/Biology/0", "kmmlu/Biology/6",
                                                             "kmmlu/Law/1", "haerae/History/0"})
    assert k["n"] == 5 and k["excluded"] == 2
    # shots=3 → 행 0~2는 예시. 행 3(채점 첫 문항)과 행 7은 빼고, 예시 행 1은 영향 없음
    h = ev.haerae_details(model, tiny_tok, 3, "cpu", exclude={"haerae/History/3", "haerae/History/7",
                                                              "haerae/History/1"})
    assert h["n"] == 3 and h["excluded"] == 2
    assert ev._skip({"kobest/copa/4", "kobest/copa/x", "kobest/boolq/1"}, "kobest/copa/") == {4}
    assert ev._skip(None, "kobest/copa/") is None
