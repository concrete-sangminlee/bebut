"""CLI 스모크 테스트 — sample.py(이어쓰기·대화)를 실제 프로세스로 돌린다."""

import os
import subprocess
import sys

import pytest
import torch

from tests.conftest import ROOT


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory, tiny_tok):
    from configs.ci_tiny import model as cfg
    from model import GPT

    d = tmp_path_factory.mktemp("cli")
    tiny_tok.save(str(d / "tok.json"))
    torch.manual_seed(0)
    torch.save({"model": GPT(cfg).state_dict(), "step": 3, "config": "ci_tiny"}, d / "m.pt")
    return d


def run(args, stdin=""):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "sample.py"), *args], input=stdin,
                       capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": ROOT}, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_completion_mode(ckpt):
    out = run(["--ckpt", str(ckpt / "m.pt"), "--tokenizer", str(ckpt / "tok.json"),
               "--prompt", "안녕하세요", "--max-tokens", "8", "--seed", "1"])
    assert out.startswith("안녕하세요")


def test_chat_mode_commands_and_long_history(ckpt):
    # ci_tiny는 문맥이 64토큰이라 몇 턴만 지나도 기록이 넘친다 — 잘라 넣으며 계속 돌아야 한다
    turns = "\n".join(["안녕하세요"] * 6 + ["/reset", "/system 너는 비서다", "질문", "/exit"]) + "\n"
    out = run(["--ckpt", str(ckpt / "m.pt"), "--tokenizer", str(ckpt / "tok.json"), "--chat",
               "--max-tokens", "16", "--temperature", "0", "--verbose"], turns)
    assert out.count("모델:") == 7
    assert "(기록을 지웠습니다)" in out and "(시스템 프롬프트: 너는 비서다)" in out


def test_default_dtype(monkeypatch):
    import sample

    assert sample.default_dtype("cuda") == torch.bfloat16
    assert sample.default_dtype("mps") == torch.float32
    monkeypatch.setattr(sample, "cpu_has_fast_bf16", lambda: True)
    assert sample.default_dtype("cpu") == torch.bfloat16
    monkeypatch.setattr(sample, "cpu_has_fast_bf16", lambda: False)
    assert sample.default_dtype("cpu") == torch.float32
