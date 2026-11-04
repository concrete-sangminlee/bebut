"""학습 루프 e2e: 실제 train.py를 서브프로세스로 돌려 재개의 정확성을 검증한다.

중단 없이 12 step 학습한 결과와, annealing 전환(step 8) 앞뒤에서 두 번 끊고 재개한
결과의 가중치가 **비트 단위로 같아야** 한다. 예전 구현은 annealing 구간에서 재개하면
데이터 스트림을 되감아서 이 검증에 실패한다.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from tests.conftest import ROOT


def run_train(*args, cwd):
    env = {**os.environ, "PYTHONPATH": ROOT}
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--config", "ci_tiny", *args]
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, tiny_tok):
    d = tmp_path_factory.mktemp("train")
    tok_path = d / "tok.json"
    tiny_tok.save(str(tok_path))
    rng = np.random.default_rng(0)
    for name, hi in (("web", 600), ("anneal", 300)):  # 두 데이터의 분포를 다르게
        (d / name).mkdir()
        for split, n in (("train", 40_000), ("val", 4_000)):
            rng.integers(0, hi, n).astype(np.uint16).tofile(d / name / f"{split}_000000.bin")
    return d, tok_path


def common(d, tok_path, out):
    return ["--shard-dir", str(d / "web"), "--anneal-shard-dir", str(d / "anneal"),
            "--tokenizer", str(tok_path), "--out-dir", str(d / out)]


def test_resume_across_annealing_is_bit_exact(workspace):
    d, tok = workspace
    log = run_train(*common(d, tok, "full"), cwd=d)
    assert "annealing 데이터로 전환" in log and "학습 완료" in log

    ck = d / "split" / "ci_tiny" / "latest.pt"
    run_train(*common(d, tok, "split"), "--stop-after", "5", cwd=d)
    run_train(*common(d, tok, "split"), "--resume", str(ck), "--stop-after", "9", cwd=d)
    log = run_train(*common(d, tok, "split"), "--resume", str(ck), cwd=d)

    a = torch.load(d / "full" / "ci_tiny" / "latest.pt", weights_only=False)
    b = torch.load(ck, weights_only=False)
    assert a["step"] == b["step"] == 11
    assert a["train_loader"] == b["train_loader"]
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    assert "annealing 구간 재개" in log

    # 스냅샷: 6의 배수 + 마지막 step + annealing 직전(7)
    snaps = sorted(os.listdir(d / "full" / "ci_tiny"))
    assert {"step_000006.pt", "step_000007.pt", "step_000011.pt"} <= set(snaps)

    # metrics.csv: 재개해도 헤더가 한 번만, step이 빠짐없이
    rows = (d / "split" / "ci_tiny" / "metrics.csv").read_text().strip().splitlines()
    assert rows[0].startswith("step,") and sum(r.startswith("step,") for r in rows) == 1
    assert [int(r.split(",")[0]) for r in rows[1:]] == list(range(12))
