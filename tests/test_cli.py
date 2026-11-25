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


def test_bf16_detection_requires_real_bf16_instructions(monkeypatch):
    """AVX-512만 있고 bf16 명령이 없는 CPU에선 fp32 — 흉내 bf16은 prefill이 몇 배 느리다."""
    import sample

    for avx512_bf16, amx, want in ((False, False, False), (True, False, True), (False, True, True)):
        monkeypatch.setattr(torch.cpu, "_is_avx512_bf16_supported", lambda v=avx512_bf16: v, raising=False)
        monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda v=amx: v, raising=False)
        assert sample.cpu_has_fast_bf16() is want


def test_bench_infer_smoke(tmp_path):
    """추론 벤치 스크립트가 작은 설정에서 끝까지 돌고 결과 형식이 맞는다 (GGUF 없이)."""
    import json

    out = tmp_path / "bench.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "bench_infer.py"), "--config", "ci_tiny", "--reps", "1",
                        "--threads", "1", "--types", "--out", str(out)], capture_output=True, text=True,
                       timeout=300, env={**os.environ, "PYTHONPATH": ROOT}, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.load(open(out, encoding="utf-8"))
    assert res["torch"]["fp32_decode_tok_s"] > 0 and res["torch"]["bf16_decode_tok_s"] > 0
    assert res["ttft"]["no_cache_s"] > 0 and res["ttft"]["prefix_cache_s"] > 0
    assert set(res["sampler_ms"]) == {"greedy", "top_p_0.95_peaked", "top_p_0.95_flat", "top_p_1.0"}
    assert res["env"]["threads"] == 1 and res["gguf"] is None
    assert res["window_reuse"] is None  # 문맥이 짧은 설정에서는 대화 모의를 하지 않는다


def test_window_reuse_simulation_uses_the_given_context():
    """문맥 창 모의: 창이 찬 뒤에도 대부분의 문맥을 캐시에서 재사용하고, 문맥 길이를 결과에 남긴다.
    예전엔 재는 설정과 무관하게 늘 2048로 모의해, 문맥 1024인 설정에 2048의 수치를 적었다."""
    import bench_infer
    from tokenizer.bpe import ByteBPETokenizer

    tok = ByteBPETokenizer.load(os.path.join(ROOT, "tokenizer", "tokenizer.json"))
    big, small = bench_infer.window_reuse(tok, 2048, 256), bench_infer.window_reuse(tok, 1024, 256)
    assert big["max_seq_len"] == 2048 and small["max_seq_len"] == 1024
    assert big["turns_measured"] > 30 and big["reused_per_turn"] > 3 * big["recomputed_per_turn"]
    assert small["reused_per_turn"] < big["reused_per_turn"] / 2  # 창이 작으면 재사용할 문맥도 작다
    assert small["reused_per_turn"] + small["recomputed_per_turn"] <= 1024 - 256
