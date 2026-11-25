"""신뢰구간·유의성 검정(evals/stats.py)과 비교 CLI."""

import json
import os
import subprocess
import sys

import pytest

import eval as ev
from evals.stats import (compare_benchmark, compare_results, half_width, mcnemar, paired_diff,
                         unpaired_diff, wilson_ci)
from tests.conftest import ROOT


def test_mcnemar_exact_values():
    assert mcnemar(0, 0) == 1.0 and mcnemar(0, 1) == 1.0  # 한 문제 차이는 아무것도 아니다
    assert mcnemar(0, 5) == pytest.approx(2 / 32)
    assert mcnemar(1, 9) == mcnemar(9, 1) == pytest.approx(22 / 1024)
    assert mcnemar(40, 40) == 1.0
    # 쌍이 많을 때의 정규 근사가 정확 계산과 이어진다
    assert mcnemar(950, 1051) == pytest.approx(mcnemar(950, 1050), rel=0.15)


def test_paired_beats_unpaired_on_the_same_items():
    """같은 500문항에서 B가 A의 오답 30개를 고치고 5개를 새로 틀렸다. 정답 수만 보면(짝 없는 비교)
    잡음과 구분되지 않지만, 문항을 맞대면 분명하다 — 그래서 평가 결과에 문항별 정오를 남긴다."""
    a = "1" * 200 + "0" * 300
    b = "1" * 195 + "0" * 5 + "1" * 30 + "0" * 270
    paired = paired_diff(a, b)
    assert paired["a_only"] == 5 and paired["b_only"] == 30 and paired["diff"] == pytest.approx(0.05)
    assert paired["p"] < 0.001 and paired["ci95"][0] > 0
    loose = unpaired_diff(200, 500, 225, 500)
    assert loose["p"] > 0.05 and loose["ci95"][0] < 0 < loose["ci95"][1]


def test_paired_ignores_unscored_and_checks_length():
    r = paired_diff("1x0-1", "0x1-1")  # x: 오염 제외, -: 채점 불가
    assert r["n"] == 3 and r["a_only"] == 1 and r["b_only"] == 1 and r["diff"] == 0
    with pytest.raises(ValueError):
        paired_diff("10", "101")


def test_half_width_and_known_case():
    assert half_width(150) == pytest.approx(0.080, abs=0.001)  # 마일스톤 평가가 쓰던 크기
    assert half_width(500) == pytest.approx(0.044, abs=0.001)
    c = unpaired_diff(231, 500, 200, 500)  # annealing 전후 HellaSwag 46.2% → 40.0% (커밋된 기록)
    assert c["diff"] == pytest.approx(-0.062) and 0.04 < c["p"] < 0.06
    assert wilson_ci(21, 30)[0] < 0.55


def test_evaluate_choices_records_items(monkeypatch):
    def fake_scores(model, tok, pairs, device, stats):
        return [5.0 if c == "정답" else 0.0 for _, c in pairs]

    monkeypatch.setattr(ev, "score_continuations", fake_scores)
    rows = [{"c": ["정답", "오답"], "l": 0}, {"c": ["오답", "정답"], "l": 0},  # 맞힘, 틀림
            {"c": ["같음", "같음"], "l": 0}, {"c": ["정답", "오답"], "l": 0}]   # 채점 불가, 오염 제외
    r = ev.evaluate_choices(None, None, lambda row: ("문맥 ", row["c"], row["l"]), [], rows, 0, "cpu", skip={3})
    assert r["items"]["acc"] == "10-x" and r["n"] == 2 and r["excluded"] == 1 and r["degenerate"] == 1
    worse = {**r, "items": {k: "00-x" for k in r["items"]}, "acc_calib": 0.0}
    c = compare_benchmark(r, worse, "acc")
    assert c["method"] == "paired" and c["n"] == 2 and c["a_only"] == 1
    # 과목별 결과(by_config)는 과목 이름순으로 이어 붙여 짝을 맞춘다
    agg = {"by_config": {"b": {"items": {"acc": "10"}}, "a": {"items": {"acc": "01"}}}}
    agg2 = {"by_config": {"a": {"items": {"acc": "11"}}, "b": {"items": {"acc": "10"}}}}
    assert compare_benchmark(agg, agg2, "acc")["b_only"] == 1


def test_old_format_logs_fall_back_to_unpaired():
    old = json.load(open(os.path.join(ROOT, "logs", "final_eval.json"), encoding="utf-8"))
    res = compare_results(old["annealing 직전"], old["최종"])
    assert set(res) == {"copa", "boolq", "hellaswag"} and res["hellaswag"]["method"] == "unpaired"
    assert res["copa"]["p"] > 0.5 and res["hellaswag"]["p"] < 0.06


def test_compare_cli_on_committed_logs(tmp_path):
    out = tmp_path / "cmp.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "compare_evals.py"),
                        os.path.join(ROOT, "logs", "sft_eval.json"), "--a", "베이스", "--b", "SFT 최종",
                        "--out", str(out)], capture_output=True, text=True, timeout=120,
                       cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr[-1500:]
    assert "짝 비교 n=30" in r.stdout and "잡음과 구분 안 됨" in r.stdout
    saved = json.load(open(out, encoding="utf-8"))["comparisons"]
    assert saved["대화"]["b_only"] == 1 and saved["대화"]["a_only"] == 0 and saved["대화"]["p"] == 1.0
    bad = subprocess.run([sys.executable, os.path.join(ROOT, "compare_evals.py"),
                          os.path.join(ROOT, "logs", "sft_eval.json"), "--a", "없는 모델", "--b", "베이스"],
                         capture_output=True, text=True, timeout=120, cwd=ROOT,
                         env={**os.environ, "PYTHONPATH": ROOT})
    assert bad.returncode != 0 and "없는 모델" in bad.stderr
