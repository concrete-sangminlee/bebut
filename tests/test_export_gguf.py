"""GGUF 내보내기 검증 — gguf 리더로 구조를, llama.cpp로 실제 추론·토큰화를 확인한다.

둘 다 저장소 의존성이 아니므로 없으면 해당 테스트만 건너뛴다.
설치: pip install gguf llama-cpp-python \\
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

llama.cpp 비교는 KV 캐시를 F32로 둔다(type_k/type_v=0). 기본값 F16 캐시는 그 자체로
logits를 ~5e-3(상대) 흔들어서, 내보내기 오류와 캐시 반올림을 구분할 수 없다.
"""

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

import export_gguf
import export_hf
from chat import render
from model import GPT, ModelConfig
from tests.conftest import ROOT

gguf = pytest.importorskip("gguf")

CFG = ModelConfig(vocab_size=65536, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)
PROMPTS = ["대한민국의 수도는 서울입니다. The quick brown fox 123", "안녕하세요 저는 학생입니다",
           "def f(x): return x ** 2"]


def _random_gpt(cfg: ModelConfig, seed: int = 0) -> GPT:
    """초기화 그대로(std 0.02)면 logits가 거의 균일해 순열 실수도 통과할 수 있다.
    가중치를 크게 흔들어 차원 배치가 틀리면 확실히 드러나게 한다 (test_export_hf.py와 같은 방식)."""
    torch.manual_seed(seed)
    model = GPT(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0, 1.0 if "emb" in name else 0.3)
            else:
                p.uniform_(0.5, 1.5)
    return model.eval()


def _full_logits(model: GPT, ids: list[int]) -> np.ndarray:
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids]), targets=torch.tensor([ids]))
    return logits[0].numpy()


@pytest.fixture(scope="module")
def model():
    return _random_gpt(CFG)


@pytest.fixture(scope="module")
def exported(tmp_path_factory, model, real_tok):
    """타입별 GGUF (실제 64K vocab 토크나이저 포함)."""
    d = tmp_path_factory.mktemp("gguf")
    paths = {}
    for q in export_gguf.QTYPES:
        paths[q] = str(d / f"m_{q}.gguf")
        export_gguf.export_gguf(model.state_dict(), CFG, real_tok, paths[q], q, chat=True,
                                name="bebut-test", write_modelfile=(q == "q8_0"))
    return paths


def _field(reader, key):
    return reader.fields[key].contents()


def _expected_tensors(model) -> dict[str, torch.Tensor]:
    hf = export_hf.convert_state_dict(model.state_dict(), CFG)
    return {export_gguf.gguf_name(k): v for k, v in hf.items()}


# ---------------- 구조 (gguf 리더) ----------------


def test_metadata(exported, real_tok):
    r = gguf.GGUFReader(exported["f32"])
    sp = real_tok.special_tokens
    expect = {
        "general.architecture": "qwen3", "general.name": "bebut-test", "general.file_type": 0,
        "general.alignment": 32,
        "qwen3.context_length": 128, "qwen3.embedding_length": 64, "qwen3.block_count": 2,
        "qwen3.feed_forward_length": CFG.ffn_dim, "qwen3.attention.head_count": 4,
        "qwen3.attention.head_count_kv": 2, "qwen3.attention.key_length": 16,
        "qwen3.attention.value_length": 16,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "llama-bpe",
        "tokenizer.ggml.bos_token_id": sp["<|bos|>"], "tokenizer.ggml.eos_token_id": sp["<|im_end|>"],
        "tokenizer.ggml.eot_token_id": sp["<|im_end|>"], "tokenizer.ggml.padding_token_id": sp["<|pad|>"],
        "tokenizer.ggml.add_bos_token": False, "tokenizer.ggml.add_eos_token": False,
        "tokenizer.chat_template": export_hf.CHAT_TEMPLATE,
    }
    for k, v in expect.items():
        assert _field(r, k) == v, k
    assert _field(r, "qwen3.rope.freq_base") == pytest.approx(CFG.rope_theta)
    assert _field(r, "qwen3.attention.layer_norm_rms_epsilon") == pytest.approx(CFG.norm_eps)
    file_types = {q: _field(gguf.GGUFReader(p), "general.file_type") for q, p in exported.items()}
    assert file_types == {"f32": 0, "f16": 1, "bf16": 32, "q8_0": 7}


def test_vocab_arrays(exported, real_tok):
    r = gguf.GGUFReader(exported["f32"])
    tokens, types = _field(r, "tokenizer.ggml.tokens"), _field(r, "tokenizer.ggml.token_type")
    merges = _field(r, "tokenizer.ggml.merges")
    b2u = export_hf.bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}
    assert len(tokens) == len(types) == real_tok.vocab_size == CFG.vocab_size
    for i in range(256 + len(real_tok.merges)):  # 문자열을 바이트로 되돌리면 bebut vocab과 같다
        assert bytes(u2b[c] for c in tokens[i]) == real_tok.vocab[i]
    for text, i in real_tok.special_tokens.items():
        assert tokens[i] == text and types[i] == export_gguf.TOKEN_CONTROL
    assert set(types[:256 + len(real_tok.merges)]) == {export_gguf.TOKEN_NORMAL}
    assert len(merges) == len(real_tok.merges)
    a, b = real_tok.merges[1000]
    assert merges[1000] == f"{tokens[a]} {tokens[b]}"


def test_tensor_layout(exported, model):
    r = gguf.GGUFReader(exported["f32"])
    expected = _expected_tensors(model)
    names = {t.name for t in r.tensors}
    assert names == set(expected) and "output.weight" not in names  # 임베딩 공유
    for t in r.tensors:
        assert list(t.shape) == list(reversed(expected[t.name].shape)), t.name  # ggml은 역순
        assert t.data_offset % 32 == 0, t.name


@pytest.mark.parametrize("qtype,np_dtype,torch_dtype", [
    ("f32", "<f4", torch.float32), ("f16", "<f2", torch.float16), ("bf16", None, torch.bfloat16)])
def test_float_data_bit_exact(exported, model, qtype, np_dtype, torch_dtype):
    r = gguf.GGUFReader(exported[qtype])
    expected = _expected_tensors(model)
    for t in r.tensors:
        src = expected[t.name]
        if src.dim() == 1:  # norm은 항상 F32
            assert t.tensor_type == gguf.GGMLQuantizationType.F32
            assert np.array_equal(np.asarray(t.data).reshape(-1), src.numpy())
            continue
        conv = src.to(torch_dtype)
        want = (conv.view(torch.int16).numpy() if torch_dtype is torch.bfloat16
                else conv.numpy().astype(np_dtype)).tobytes()
        assert np.asarray(t.data).tobytes() == want, t.name


def test_q8_0_matches_reference_quantizer(exported, model):
    """gguf-py에 들어 있는 llama.cpp 참조 구현과 바이트까지 같아야 한다."""
    r = gguf.GGUFReader(exported["q8_0"])
    expected = _expected_tensors(model)
    for t in r.tensors:
        src = expected[t.name].float().numpy()
        if src.ndim == 1:
            assert t.tensor_type == gguf.GGMLQuantizationType.F32
            continue
        assert t.tensor_type == gguf.GGMLQuantizationType.Q8_0, t.name
        ref = gguf.quants.quantize(src, gguf.GGMLQuantizationType.Q8_0)
        assert np.asarray(t.data).tobytes() == ref.tobytes(), t.name


def test_q8_0_quantizer_edge_cases():
    rng = np.random.default_rng(0)
    for scale in (1e-3, 1.0, 1e3):
        x = (rng.standard_normal((8, 96)) * scale).astype(np.float32)
        x[0, :32] = 0.0  # 전부 0인 블록 — 0으로 나누면 안 된다
        raw = export_gguf.quantize_q8_0(x)
        assert raw == gguf.quants.quantize(x, gguf.GGMLQuantizationType.Q8_0).tobytes()
        deq = export_gguf.dequantize_q8_0(raw, x.shape)
        d = np.abs(x.reshape(-1, 32)).max(1) / 127
        d16 = d.astype(np.float16).astype(np.float32)
        bound = 0.5 * d16 + 127 * np.abs(d16 - d) + 1e-12  # 반 칸 + 스케일의 fp16 반올림
        assert (np.abs(deq - x).reshape(-1, 32).max(1) <= bound * 1.0001).all()
        assert np.all(deq[0, :32] == 0)
    # 0.5 바로 아래(0.49999997)는 roundf로 0 — 예전 floor(|v| + 0.5)는 float32 반올림으로 1을 냈다
    x = np.zeros((1, 32), dtype=np.float32)
    x[0, 0], x[0, 1] = 127.0, np.nextafter(np.float32(0.5), np.float32(0))
    assert export_gguf.quantize_q8_0(x) == gguf.quants.quantize(x, gguf.GGMLQuantizationType.Q8_0).tobytes()


def test_q8_0_falls_back_for_unaligned_rows(tmp_path, real_tok):
    """행 길이가 32의 배수가 아니면 Q8_0 블록을 못 만들어 F16으로 둔다."""
    cfg = ModelConfig(vocab_size=65536, dim=48, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32)
    p = str(tmp_path / "odd.gguf")
    export_gguf.export_gguf(_random_gpt(cfg).state_dict(), cfg, real_tok, p, "q8_0",
                            write_modelfile=False)
    types = {t.name: t.tensor_type.name for t in gguf.GGUFReader(p).tensors}
    assert types["blk.0.attn_q.weight"] == "F16"       # 입력 차원 48
    assert types["blk.0.ffn_down.weight"] == "Q8_0"    # 입력 차원 ffn_dim=128
    assert types["blk.0.attn_norm.weight"] == "F32"


def test_small_tokenizer_is_padded_to_model_vocab(tmp_path, tiny_tok):
    cfg = ModelConfig(vocab_size=1024, dim=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32)
    p = str(tmp_path / "pad.gguf")
    export_gguf.export_gguf(_random_gpt(cfg).state_dict(), cfg, tiny_tok, p, "f32",
                            write_modelfile=False)
    r = gguf.GGUFReader(p)
    tokens, types = _field(r, "tokenizer.ggml.tokens"), _field(r, "tokenizer.ggml.token_type")
    assert len(tokens) == 1024 and tokens[-1] == "<|unused_1023|>"
    assert types[tiny_tok.vocab_size:] == [export_gguf.TOKEN_UNUSED] * (1024 - tiny_tok.vocab_size)


def test_chat_mode_detection():
    for name in ("sft_1b", "dpo_1b", "ci_tiny_sft", "ci_tiny_dpo", "sft_smoke"):
        assert export_gguf.is_chat_config(name), name
    for name in ("base_1b", "ci_tiny", "mid_125m", "debug_30m", "smoke"):
        assert not export_gguf.is_chat_config(name), name


def test_unknown_tensor_is_rejected():
    with pytest.raises(ValueError):
        export_gguf.gguf_name("model.layers.0.mlp.experts.weight")


# ---------------- Ollama Modelfile ----------------


def _render_ollama(template: str, messages) -> str:
    """Go 템플릿 중 이 Modelfile이 쓰는 부분({{- range .Messages }}…{{ end }})만 흉내 낸다."""
    head, rest = template.split("{{- range .Messages }}")
    body, tail = rest.split("{{ end }}")
    out = head
    for m in messages:
        out += body.replace("{{ .Role }}", m["role"]).replace("{{ .Content }}", m["content"])
    return out + tail


def test_modelfile(exported):
    path = os.path.join(os.path.dirname(exported["q8_0"]), "Modelfile")
    text = open(path, encoding="utf-8").read()
    assert "FROM ./m_q8_0.gguf\n" in text
    assert 'PARAMETER stop "<|im_end|>"' in text and 'PARAMETER stop "<|eos|>"' in text
    assert "PARAMETER num_ctx 128" in text
    template = text.split('TEMPLATE """')[1].split('"""')[0]
    msgs = [{"role": "system", "content": "너는 비서다."}, {"role": "user", "content": "안녕?"},
            {"role": "assistant", "content": "안녕하세요."}, {"role": "user", "content": "이름은?"}]
    assert _render_ollama(template, msgs) == render(msgs, add_generation_prompt=True)


# ---------------- llama.cpp 실제 추론 ----------------


@pytest.fixture(scope="module")
def llama_cpp():
    return pytest.importorskip("llama_cpp")


def _llama(llama_cpp, path, **kw):
    return llama_cpp.Llama(model_path=path, n_ctx=128, logits_all=True, verbose=False, n_threads=4,
                           type_k=0, type_v=0, **kw)  # 0 = GGML_TYPE_F32 (KV 캐시)


def _llama_logits(llm, ids):
    llm.reset()
    llm.eval(ids)
    return np.array(llm.scores[: len(ids)])


def _llama_greedy(llm, ids, n):
    llm.reset()
    llm.eval(ids)
    out = []
    for _ in range(n):
        nxt = int(np.argmax(llm.scores[llm.n_tokens - 1]))
        out.append(nxt)
        llm.eval([nxt])
    return out


def test_llama_cpp_f32_logits_and_greedy_match(llama_cpp, exported, model, real_tok):
    llm = _llama(llama_cpp, exported["f32"])
    for text in PROMPTS:
        ids = real_tok.encode(text)
        ref = _full_logits(model, ids)
        diff = np.abs(_llama_logits(llm, ids) - ref).max()
        assert diff < 1e-3 * ref.std(), (text, diff, ref.std())
        greedy = model.generate(torch.tensor([ids]), 8, temperature=0.0)[0, len(ids):].tolist()
        assert _llama_greedy(llm, ids, 8) == greedy


def test_llama_cpp_needs_rope_permutation(llama_cpp, tmp_path, model, real_tok, monkeypatch):
    monkeypatch.setattr(export_hf, "rope_permutation", lambda hd: torch.arange(hd))
    p = str(tmp_path / "noperm.gguf")
    export_gguf.export_gguf(model.state_dict(), CFG, real_tok, p, "f32", write_modelfile=False)
    ids = real_tok.encode(PROMPTS[0])
    ref = _full_logits(model, ids)
    assert np.abs(_llama_logits(_llama(llama_cpp, p), ids) - ref).max() > 0.5 * ref.std()


def test_llama_cpp_half_and_q8_0_are_close(llama_cpp, exported, model, real_tok):
    """F16/BF16/Q8_0은 반올림만큼 다르다. Q8_0은 llama.cpp가 활성값도 8비트로 바꿔 곱해서
    가장 크게 다르다 — 이 테스트의 가중치는 일부러 크게 흔든 것이라 실제 모델보다 오차가 크다."""
    tol = {"f16": 0.02, "bf16": 0.15, "q8_0": 0.35}  # max|Δlogit| / std(logit)
    for q, t in tol.items():
        llm = _llama(llama_cpp, exported[q])
        agree, greedy_ok = [], 0
        for text in PROMPTS:
            ids = real_tok.encode(text)
            ref = _full_logits(model, ids)
            lc = _llama_logits(llm, ids)
            assert np.abs(lc - ref).max() < t * ref.std(), (q, text)
            agree.append((lc.argmax(-1) == ref.argmax(-1)).mean())
            g = model.generate(torch.tensor([ids]), 8, temperature=0.0)[0, len(ids):].tolist()
            greedy_ok += _llama_greedy(llm, ids, 8) == g
        assert np.mean(agree) > 0.7, (q, agree)
        assert greedy_ok >= (3 if q == "f16" else 1), (q, greedy_ok)


def _tokenizer_corpus() -> list[str]:
    texts = [
        "대한민국은 동아시아의 한반도 남부에 있는 나라이다. 수도는 서울특별시이며, 국화는 무궁화이다.",
        "The quick brown fox jumps over the lazy dog. It's I'LL we've they're don't",
        "숫자 1234567 과 3.14159, 1,000,000원, 2026-11-04 1234567890123",
        "이모지 🎉🔥👍🏽 👨‍👩‍👧 漢字 かな カタカナ 中文测试",
        "  공백   과\n\n\n줄바꿈\t\t탭  \r\n 처리  ", "a\n\n b", "x \n\n  y", "   \n", "  12 34  ",
        " 앞 공백", "뒤 공백 ", " ", "\n", "\t\n \n",
        "https://github.com/concrete-sangminlee/bebut?tab=readme#전체-재현-절차",
        "def f(x):\n    return x ** 2  # 주석\n\nprint(f(3))",
        '<html><body class="a">안녕</body></html>',
        "edge \x1c\x1d\x1e\x1f \x85 \xa0   　 ​ ﻿ ­ end",
        "é café 한가 ㄱㄴㄷ ㅏㅑ",
        "!!!???... ---- ==== ****", "'s 's'S 'Ll", "a'b'c", "x\r\ny\rz", "Ⅻ ½ ² ٣ ๓ ①②",
    ]
    rng = random.Random(0)
    pool = list("가나다라 abcXYZ 012 .,!?'\"\n\t\r ") + ["　", "\xa0", "\x85", "é", "ß", "🎉", "漢", "ㅋ"]
    texts += ["".join(rng.choice(pool) for _ in range(rng.randint(1, 60))) for _ in range(300)]
    with open(os.path.join(ROOT, "evals", "heldout_ko.json"), encoding="utf-8") as f:
        texts += [q["q"] for q in json.load(f)]
    return texts


def test_llama_cpp_tokenizer_matches_bebut(llama_cpp, exported, real_tok):
    """pre-tokenizer 'llama-bpe'(Llama 3 정규식)가 bebut의 GPT-4식 분할과 같은지 실제로 확인."""
    llm = llama_cpp.Llama(model_path=exported["f32"], vocab_only=True, verbose=False)
    bad = []
    for text in _tokenizer_corpus():
        ids = llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
        if ids != real_tok.encode(text):
            bad.append(text)
        elif llm.detokenize(ids).decode("utf-8") != text:
            bad.append(("detokenize", text))
    assert not bad, bad[:5]


def test_llama_cpp_special_tokens_and_eog(llama_cpp, exported, real_tok):
    llm = llama_cpp.Llama(model_path=exported["f32"], vocab_only=True, verbose=False)
    s = render([{"role": "user", "content": "안녕"}])
    assert llm.tokenize(s.encode(), add_bos=False, special=True) == real_tok.encode(s, allowed_special=True)
    plain = llm.tokenize(s.encode(), add_bos=False, special=False)  # 사용자 입력은 special로 해석 안 함
    assert not set(plain) & set(real_tok.special_tokens.values())
    assert llm.token_eos() == real_tok.special_tokens["<|im_end|>"]
    assert llm.token_bos() == real_tok.special_tokens["<|bos|>"]
    # 대화 모드에서도 <|eos|>가 생성 끝이어야 한다 (다른 추론 경로는 둘 다에서 멈춘다)
    vocab = llama_cpp.llama_model_get_vocab(llm._model.model)
    for t in ("<|im_end|>", "<|eos|>"):
        assert llama_cpp.llama_vocab_is_eog(vocab, real_tok.special_tokens[t]), t


# ---------------- CLI ----------------


def test_cli_from_checkpoint(tmp_path, tiny_tok):
    from configs.ci_tiny import model as cfg

    tiny_tok.save(str(tmp_path / "tok.json"))
    torch.manual_seed(0)
    m = GPT(cfg)
    torch.save({"model": m.state_dict(), "step": 7, "config": "ci_tiny"}, tmp_path / "m.pt")
    out = tmp_path / "out" / "tiny.gguf"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "export_gguf.py"), "--ckpt", str(tmp_path / "m.pt"),
                        "--out", str(out), "--tokenizer", str(tmp_path / "tok.json"), "--type", "q8_0"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "out" / "Modelfile").exists()
    reader = gguf.GGUFReader(str(out))
    assert _field(reader, "general.name") == "bebut-ci_tiny"
    assert _field(reader, "tokenizer.ggml.eos_token_id") == tiny_tok.special_tokens["<|eos|>"]  # 베이스
    try:
        import llama_cpp
    except ImportError:
        return
    llm = llama_cpp.Llama(model_path=str(out), n_ctx=64, logits_all=True, verbose=False)
    ids = tiny_tok.encode("안녕하세요")
    llm.eval(ids)
    ref = _full_logits(m.eval(), ids)  # 실제 초기화(std 0.02) 그대로인 모델도 Q8_0으로 맞는다
    lc = np.array(llm.scores[: len(ids)])
    assert np.isfinite(lc).all() and np.abs(lc - ref).max() < 0.05 * ref.std() + 1e-3
