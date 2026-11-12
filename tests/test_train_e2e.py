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
    assert "step_000000.pt" not in snaps  # step 0은 남기지 않는다 (1.2B면 14GB)
    assert not [f for f in snaps if f.endswith(".tmp")]  # 원자적 저장: 임시 파일이 남지 않는다
    last = torch.load(d / "full" / "ci_tiny" / "step_000011.pt", weights_only=False)
    assert last["step"] == 11 and all(torch.equal(last["model"][k], a["model"][k]) for k in a["model"])
    s7 = torch.load(d / "full" / "ci_tiny" / "step_000007.pt", weights_only=False)
    assert s7["step"] == 7  # 하드링크한 스냅샷은 뒤의 latest.pt 교체에 영향받지 않는다

    # metrics.csv: 재개해도 헤더가 한 번만, step이 빠짐없이
    rows = (d / "split" / "ci_tiny" / "metrics.csv").read_text().strip().splitlines()
    assert rows[0].startswith("step,") and sum(r.startswith("step,") for r in rows) == 1
    assert [int(r.split(",")[0]) for r in rows[1:]] == list(range(12))


def run_train_raw(*args, cwd):
    env = {**os.environ, "PYTHONPATH": ROOT}
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--config", "ci_tiny", *args]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def test_annealing_checkpoint_without_anneal_dir_is_refused(workspace):
    d, tok = workspace
    base = ["--shard-dir", str(d / "web"), "--tokenizer", str(tok), "--out-dir", str(d / "mm")]
    run_train(*base, "--anneal-shard-dir", str(d / "anneal"), "--stop-after", "9", cwd=d)
    r = run_train_raw(*base, "--resume", str(d / "mm" / "ci_tiny" / "latest.pt"), cwd=d)
    assert r.returncode != 0 and "annealing=True" in (r.stdout + r.stderr)


def test_headerless_legacy_metrics_csv(workspace):
    """예전 train.py가 --resume으로 처음 만든 metrics.csv엔 헤더가 없다."""
    from train import LEGACY_FIELDS, open_metrics

    path = workspace[0] / "legacy_metrics.csv"
    path.write_text("100,3.000e-04,2.5000,,\n")
    f, w = open_metrics(str(path))
    w.writerow({"step": 101, "lr": "3.000e-04", "train_loss": "2.4", "val_loss": "",
                "tokens_per_sec": "", "grad_norm": "1.0"})
    f.close()
    rows = path.read_text().splitlines()
    assert rows == ["100,3.000e-04,2.5000,,", "101,3.000e-04,2.4,,"]
    assert len(LEGACY_FIELDS) == 5


def test_metrics_rows_after_resume_point_are_dropped(tmp_path):
    """크래시로 죽으면 마지막 체크포인트 뒤 step의 행이 이미 파일에 있다. 재개는 체크포인트
    다음 step부터 다시 쓰므로 예전엔 그 구간이 두 번 들어갔다."""
    from train import METRIC_FIELDS, open_metrics

    path = tmp_path / "metrics.csv"
    path.write_text(",".join(METRIC_FIELDS) + "\n" + "".join(f"{i},1e-4,2.0,,,1.0\n" for i in range(16)))
    f, w = open_metrics(str(path), start_step=12)  # 체크포인트는 step 11
    for i in range(12, 14):
        w.writerow({"step": i, "lr": "1e-4", "train_loss": "1.9", "val_loss": "",
                    "tokens_per_sec": "", "grad_norm": "1.0"})
    f.close()
    steps = [int(r.split(",")[0]) for r in path.read_text().splitlines()[1:]]
    assert steps == list(range(14))


def test_step_if_finite_skips_nan_and_gives_up_after_limit():
    """train.py · sft.py · dpo.py 공용 가드. 예전 sft.py·dpo.py엔 없어서 NaN 한 번이면 가중치가
    전부 NaN이 된 채 끝까지 돌아 latest.pt를 덮어쓰고 '완료'로 끝났다."""
    from train import MAX_BAD_STEPS, step_if_finite

    w = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.SGD([w], lr=0.1)
    w.grad = torch.full((3,), float("nan"))
    bad = step_if_finite(opt, float("nan"), torch.tensor(float("nan")), 0, step=1)
    assert bad == 1 and torch.equal(w.data, torch.ones(3))  # 건너뛰었다
    w.grad = torch.ones(3)
    assert step_if_finite(opt, 1.0, torch.tensor(1.0), bad, step=2) == 0 and w.data[0] < 1
    with pytest.raises(RuntimeError):
        for i in range(MAX_BAD_STEPS):
            bad = step_if_finite(opt, float("inf"), torch.tensor(1.0), bad, step=3 + i)


def test_device_type_of_handles_indexed_cuda():
    from sample import device_type_of

    assert [device_type_of(d) for d in ("cuda:1", "cuda", "cpu", "mps")] == ["cuda", "cuda", "cpu", "mps"]
