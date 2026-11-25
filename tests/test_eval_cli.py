"""평가 스크립트(eval.py · final_eval.py · sft_eval.py)를 실제 프로세스로 끝까지 돌린다.

RESULTS §0은 †로 표시한 수치를 이 세 스크립트로 다시 재라고 안내하는데, 지금까지 테스트는
함수만 불렀고 main()은 한 번도 돌리지 않았다 — 그래서 없는 기본 체크포인트 경로, 커밋된 기준값
덮어쓰기 같은 문제가 그대로 있었다. 여기서는 초소형 체크포인트와 가짜 datasets 모듈(PYTHONPATH
맨 앞)로 CPU에서 몇 초 만에 돌린다.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

from tests.conftest import ROOT

FAKE_DATASETS = '''
"""테스트용 가짜 datasets — 평가 스크립트가 쓰는 만큼만."""


class Rows(list):
    def __getitem__(self, i):
        r = super().__getitem__(i)
        return Rows(r) if isinstance(i, slice) else r

    def select(self, idx):
        return Rows(self[i] for i in idx)


def _kmmlu(i):
    return {"question": f"다음 중 {i}번 물질의 특성이 아닌 것은?", "A": "환원당", "B": "알도오스",
            "C": "단당류", "D": f"{i}탄당", "answer": 1 + i % 4}


def _haerae(i):
    return {"question": f"{i}번 낱말의 뜻으로 알맞은 것은?", "a": "고인돌", "b": "돌무지무덤", "c": "널무덤",
            "d": "독무덤", "e": "움집", "answer": "abcde"[i % 5]}


DATA = {
    "skt/kobest_v1": {
        "copa": {s: Rows({"premise": f"비가 {i}시간 내렸다.", "question": "결과" if i % 2 else "원인",
                          "alternative_1": "땅이 젖었다.", "alternative_2": "해가 떴다.", "label": i % 2}
                         for i in range(8)) for s in ("train", "test")},
        "boolq": {s: Rows({"paragraph": f"서울은 대한민국의 수도이다. 인구는 약 {i}백만 명이다.",
                           "question": "서울은 수도인가?", "label": i % 2} for i in range(8))
                  for s in ("train", "test")},
        "hellaswag": {s: Rows({"context": f"남자가 {i}층 계단을 올라갔다.", "ending_1": "문을 열었다.",
                               "ending_2": "잠을 잤다.", "ending_3": "밥을 먹었다.", "ending_4": "책을 읽었다.",
                               "label": i % 4} for i in range(8)) for s in ("train", "test")},
    },
    "HAERAE-HUB/KMMLU": {sub: {"dev": Rows(_kmmlu(i) for i in range(5)), "test": Rows(_kmmlu(i) for i in range(6))}
                         for sub in ("Biology", "Law")},
    "HAERAE-HUB/HAE_RAE_BENCH_1.0": {"History": {"test": Rows(_haerae(i) for i in range(9))}},
    "HuggingFaceFW/fineweb-2": {"kor_Hang": {"test": Rows(
        {"text": f"{i}번째 글. " + "한글은 조선의 세종이 만든 글자이고, 서울은 대한민국의 수도이다. " * (20 + 30 * i)}
        for i in range(4))}},
}


def get_dataset_config_names(name):
    return list(DATA[name])


def load_dataset(name, config=None, split=None, **kw):
    d = DATA[name][config]
    return d[split] if split else d
'''


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    """초소형 체크포인트 둘(가중치가 다르다) + 가짜 datasets + 오염 id 목록 + 대화 문항."""
    from configs.ci_eval import model as cfg
    from model import GPT

    d = tmp_path_factory.mktemp("evalcli")
    (d / "stub").mkdir()
    (d / "stub" / "datasets.py").write_text(FAKE_DATASETS, encoding="utf-8")
    for name, seed in (("a.pt", 0), ("b.pt", 1)):
        torch.manual_seed(seed)
        torch.save({"model": GPT(cfg).state_dict(), "step": 7 + seed, "config": "ci_eval"}, d / name)
    (d / "ids.json").write_text(json.dumps(["kmmlu/Biology/1", "kobest/copa/0", "haerae/History/4", "heldout/1"]))
    (d / "q.json").write_text(json.dumps([
        {"type": "사실", "q": "대한민국의 수도는?", "a": ["서울"]},
        {"type": "사실", "q": "태양계에서 가장 큰 행성은?", "a": ["목성"]},
        {"type": "지시", "q": "인사해줘", "a": []}], ensure_ascii=False), encoding="utf-8")
    return d


def run(ws, script, *args):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ws / "stub"), ROOT])}
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), *args], cwd=ws, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    return r.stdout


TOK = os.path.join(ROOT, "tokenizer", "tokenizer.json")


def test_eval_py_cli(ws):
    out = run(ws, "eval.py", "--ckpt", "a.pt", "--tokenizer", TOK, "--bench", "kmmlu", "--limit", "4",
              "--exclude-ids", "ids.json")
    assert "kmmlu (cloze, 5-shot, n=7" in out and "오염 제외 1" in out and "Biology" in out
    out = run(ws, "eval.py", "--ckpt", "a.pt", "--tokenizer", TOK, "--bench", "haerae", "--format", "mc",
              "--shots", "3", "--limit", "4")
    assert "haerae (mc, 3-shot, n=4" in out
    out = run(ws, "eval.py", "--ckpt", "a.pt", "--tokenizer", TOK, "--kobest", "boolq", "--shots", "2", "--limit", "6")
    assert "KOBEST boolq (2-shot, n=6" in out and "acc_calib" in out


def test_final_eval_cli_writes_items_and_compares_checkpoints(ws):
    out = run(ws, "final_eval.py", "--ckpts", "없는것=missing.pt", "직전=a.pt", "최종=b.pt", "--tokenizer", TOK,
              "--skip-ppl", "--limit", "4", "--exclude-ids", "ids.json", "--out", "final.json")
    assert "건너뜀: 없는것" in out            # 체크포인트 하나가 없어도 나머지를 잰다
    assert "비교: 직전 → 최종" in out and "짝 비교" in out  # 같은 문항 짝 비교가 함께 나온다
    res = json.load(open(ws / "final.json", encoding="utf-8"))
    assert list(res) == ["직전", "최종"] and res["직전"]["step"] == 7
    copa = res["최종"]["kobest"]["copa"]
    assert copa["n"] == 3 and copa["excluded"] == 1 and copa["items"]["acc_calib"][0] == "x"
    assert len(copa["items"]["acc"]) == 4 and len(copa["ci95"]["acc_calib"]) == 2
    kmmlu = res["최종"]["bench"]["kmmlu"]
    assert kmmlu["n"] == 7 and kmmlu["excluded"] == 1 and set(kmmlu["by_config"]) == {"Biology", "Law"}
    assert kmmlu["by_config"]["Biology"]["items"]["acc_calib"][1] == "x"
    assert res["최종"]["bench"]["haerae"]["n"] == 4 and len(res["최종"]["facts"]) == 4
    assert not os.path.exists(ws / "logs")    # 기본 출력 경로의 기준값 파일을 건드리지 않았다


def test_sft_eval_cli(ws):
    out = run(ws, "sft_eval.py", "--models", "베이스=a.pt", "SFT=b.pt", "--tokenizer", TOK, "--questions", "q.json",
              "--max-new", "6", "--benches", "copa", "kmmlu", "--limit", "4", "--exclude-ids", "ids.json",
              "--out", "sft.json", "--answers-out", "answers.md")
    assert "비교: 베이스 → SFT" in out
    res = json.load(open(ws / "sft.json", encoding="utf-8"))
    r = res["SFT"]
    assert r["n_scored"] == 1 and r["excluded"] == 1 and len(r["answers"]) == 3  # heldout/1은 빼고 채점
    assert r["answers"][1]["excluded"] and r["answers"][2]["correct"] is None
    assert len(r["heldout_ci95"]) == 2 and r["kobest"]["copa"]["n"] == 3 and r["bench"]["kmmlu"]["n"] == 7
    assert "## SFT" in (ws / "answers.md").read_text(encoding="utf-8")
    # 저장된 답변을 모델 없이 다시 채점해도 같은 점수
    out = run(ws, "sft_eval.py", "--rescore", "sft.json", "--questions", "q.json", "--exclude-ids", "ids.json",
              "--out", "rescored.json")
    again = json.load(open(ws / "rescored.json", encoding="utf-8"))
    assert again["SFT"]["heldout_acc"] == r["heldout_acc"] and again["베이스"]["excluded"] == 1


def test_eval_milestones_keeps_per_item_results_for_paired_comparison(ws):
    """마일스톤 평가가 step별로 문항별 정오를 남기고, compare_evals.py가 두 step을 짝 비교한다.
    예전엔 CSV의 백분율만 남아 단계 사이의 변화를 표본 오차와 구분할 수 없었다 (RESULTS §1)."""
    import numpy as np

    ckpts = ws / "checkpoints" / "ci_eval"
    ckpts.mkdir(parents=True)
    for src, step in (("a.pt", 4000), ("b.pt", 8000)):
        os.link(ws / src, ckpts / f"step_{step:06d}.pt")
    (ws / "shards").mkdir()
    np.random.default_rng(0).integers(0, 60000, 30000).astype(np.uint16).tofile(ws / "shards" / "val_000000.bin")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ws / "stub"), ROOT]), "EVAL_THREADS": "2"}
    cmd = [sys.executable, os.path.join(ROOT, "eval_milestones.py"), "--config", "ci_eval", "--tokenizer", TOK,
           "--shard-dir", "shards", "--limit", "6", "--shots", "2", "--ppl-iters", "2"]
    r = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    doc = json.load(open(ckpts / "milestones.json", encoding="utf-8"))
    assert list(doc) == ["step 4000", "step 8000"] and doc["step 8000"]["limit"] == 6
    assert len(doc["step 4000"]["kobest"]["boolq"]["items"]["acc_calib"]) == 6
    assert len(open(ckpts / "milestones.csv").read().splitlines()) == 3  # 머리글 + 2행 (기존 형식 그대로)
    out = run(ws, "compare_evals.py", "checkpoints/ci_eval/milestones.json", "--a", "step 4000", "--b", "step 8000")
    assert out.count("짝 비교 n=6") == 3  # 세 과제 모두 같은 문항끼리
    # 다시 부르면 이미 평가한 step은 건너뛴다 (JSON도 그대로)
    r = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "step   4000" not in r.stdout
    assert json.load(open(ckpts / "milestones.json", encoding="utf-8")) == doc


def test_remeasure_script_end_to_end(ws, tmp_path):
    """remeasure.sh를 초소형 체크포인트로 끝까지 돌린다. 단계마다 라벨·파일 이름이 다음 단계와 맞물려야
    한다 — 어긋나면 마지막의 짝 비교(compare_evals.py)가 라벨을 못 찾아 실패한다."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ws / "stub"), ROOT]), "PY": sys.executable,
           "OUT": str(tmp_path), "PRE": str(ws / "a.pt"), "BASE": str(ws / "b.pt"), "SFT_E0": str(ws / "a.pt"),
           "SFT": str(ws / "b.pt"), "DEVICE": "cpu", "LIMIT": "4"}
    env.pop("DRY_RUN", None)

    def remeasure(**extra):
        r = subprocess.run(["bash", os.path.join(ROOT, "remeasure.sh")], env={**env, **extra}, capture_output=True,
                           text=True, timeout=900)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return r.stdout

    dry = remeasure(DRY_RUN="1")
    assert all(x in dry for x in ("final_eval.py", "sft_eval.py", "evals.text_bpb", "evals.quant_quality",
                                  "compare_evals.py", "--limit 4"))
    assert not os.listdir(tmp_path)  # 명령만 보여주고 아무것도 돌리지 않았다

    out = remeasure()
    final = json.load(open(tmp_path / "final_eval_v2.json", encoding="utf-8"))
    assert list(final) == ["annealing 직전", "최종"] and final["최종"]["bench"]["kmmlu"]["n"] > 0
    sft = json.load(open(tmp_path / "sft_eval_v2.json", encoding="utf-8"))
    assert list(sft) == ["베이스", "SFT 에폭0", "SFT 최종"] and (tmp_path / "sft_eval_answers_v2.md").exists()
    bpb = json.load(open(tmp_path / "text_bpb_bebut.json", encoding="utf-8"))
    assert list(bpb["models"]) == ["annealing 직전", "최종", "SFT 최종"]
    assert all(m["domains"]["웹 (FineWeb-2 ko test)"]["bpb"] > 0 for m in bpb["models"].values())
    # 가짜 글은 커밋된 공개 모델 기준값(logs/text_bpb.json)과 다른 글이라 한 표에 넣지 않는다
    assert ("합치지 않음" in out) == os.path.exists(os.path.join(ROOT, "logs", "text_bpb.json"))
    assert "A = annealing 직전  →  B = 최종" in out and "A = 베이스  →  B = SFT 최종" in out
    assert out.count("짝 비교 n=") >= 8  # 두 비교 모두 문항별 기록으로 짝을 지었다
    try:
        import llama_cpp  # noqa: F401
        assert set(json.load(open(tmp_path / "quant_quality_bebut.json", encoding="utf-8"))["results"]) == {
            "torch_fp32", "f16", "q8_0", "q4_0"}
    except ImportError:
        assert "llama-cpp-python이 없다" in out and "건너뛴 단계" in out

    again = remeasure()  # 결과가 있으면 다시 재지 않는다
    assert again.count("[건너뜀]") >= 3 and "=====" not in again

    missing = remeasure(OUT=str(tmp_path / "none"), BASE=str(ws / "no.pt"), SFT=str(ws / "no2.pt"))
    assert "체크포인트가 없어 건너뛴다" in missing and "건너뛴 단계" in missing
