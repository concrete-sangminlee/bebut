"""HF(Qwen3) 내보내기 검증 — transformers로 실제로 읽어서 bebut과 같은 결과가 나오는지 본다.

transformers는 저장소 의존성이 아니므로 없으면 통째로 건너뛴다.
설치: pip install transformers  (tokenizers·safetensors·jinja2가 함께 설치된다)
"""

import json
import os
import random
import subprocess
import sys

import pytest
import torch

import export_hf
from model import GPT, ModelConfig
from tokenizer.bpe import ByteBPETokenizer

transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.path.join(ROOT, "tokenizer", "tokenizer.json")


def _random_gpt(cfg: ModelConfig, seed: int = 0) -> GPT:
    """초기화 그대로(std 0.02)면 logits가 거의 균일해 순열 실수도 통과할 수 있다.
    가중치를 크게 흔들어 차원 배치가 틀리면 확실히 드러나게 한다."""
    torch.manual_seed(seed)
    model = GPT(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0, 1.0 if "emb" in name else 0.3)
            else:
                p.uniform_(0.5, 1.5)  # norm 가중치도 1이 아니어야 q/k_norm 순열이 검증된다
    return model.eval()


def _full_logits(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    # targets를 주면 전 위치 logits를 돌려준다 (targets 없으면 마지막 위치만)
    with torch.no_grad():
        return model(idx, targets=idx)[0]


@pytest.fixture(scope="module")
def bebut_tok():
    return ByteBPETokenizer.load(TOKENIZER)


@pytest.fixture(scope="module")
def exported_real_tok(tmp_path_factory, bebut_tok):
    """실제 64K 토크나이저 + vocab만 실제 크기인 초소형 모델을 내보낸 디렉토리."""
    cfg = ModelConfig(vocab_size=bebut_tok.vocab_size, dim=32, n_layers=1, n_heads=2,
                      n_kv_heads=1, max_seq_len=64)
    out = tmp_path_factory.mktemp("hf_real_tok")
    export_hf.export(_random_gpt(cfg).state_dict(), cfg, bebut_tok, str(out), "fp32", chat=True)
    return out


@pytest.fixture(scope="module")
def hf_tok(exported_real_tok):
    return transformers.AutoTokenizer.from_pretrained(str(exported_real_tok))


# ---------------- 모델 ----------------


def test_logits_and_greedy_match(tmp_path):
    cfg = ModelConfig(vocab_size=512, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)
    model = _random_gpt(cfg)
    export_hf.export(model.state_dict(), cfg, None, str(tmp_path), "fp32")
    hf, info = transformers.AutoModelForCausalLM.from_pretrained(
        str(tmp_path), dtype=torch.float32, output_loading_info=True)
    hf.eval()
    assert not info["missing_keys"] and not info["unexpected_keys"], info

    idx = torch.randint(0, cfg.vocab_size, (2, 60), generator=torch.Generator().manual_seed(1))
    ours = _full_logits(model, idx)
    with torch.no_grad():
        theirs = hf(idx).logits
    assert ours.std() > 1.0  # 의미 있는 비교인지 (균일 분포면 무엇이든 일치한다)
    diff = (ours - theirs).abs().max().item()
    assert diff < 1e-4, f"logits 불일치 {diff}"

    # greedy 생성 — HF의 KV 캐시 경로에서도 위치 인코딩이 맞는지
    prompt = idx[:, :8]
    out = prompt.clone()
    for _ in range(10):
        out = torch.cat([out, _full_logits(model, out)[:, -1].argmax(-1, keepdim=True)], dim=1)
    gen = hf.generate(prompt, max_new_tokens=10, do_sample=False, eos_token_id=None, pad_token_id=0)
    assert torch.equal(gen, out)


def test_rope_permutation_is_needed(tmp_path, monkeypatch):
    """순열 없이 내보내면 확실히 틀려야 한다 — 위 테스트가 우연히 통과하는 게 아님을 보인다."""
    cfg = ModelConfig(vocab_size=256, dim=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=64)
    model = _random_gpt(cfg)
    monkeypatch.setattr(export_hf, "rope_permutation", lambda hd: torch.arange(hd))
    export_hf.export(model.state_dict(), cfg, None, str(tmp_path), "fp32")
    hf = transformers.AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.float32).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        assert (_full_logits(model, idx) - hf(idx).logits).abs().max() > 1.0


def test_unknown_weight_is_rejected():
    cfg = ModelConfig(vocab_size=256, dim=32, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16)
    sd = GPT(cfg).state_dict()
    sd["blocks.0.attn.new_thing.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="변환 규칙"):
        export_hf.convert_state_dict(sd, cfg)


def test_config_and_generation_config(exported_real_tok, bebut_tok):
    sp = bebut_tok.special_tokens
    conf = json.loads((exported_real_tok / "config.json").read_text())
    assert conf["architectures"] == ["Qwen3ForCausalLM"]
    assert conf["tie_word_embeddings"] is True
    assert conf["eos_token_id"] == sp["<|im_end|>"]
    assert conf["pad_token_id"] == sp["<|pad|>"]
    gen = json.loads((exported_real_tok / "generation_config.json").read_text())
    assert gen["eos_token_id"] == [sp["<|im_end|>"], sp["<|eos|>"]]
    assert (exported_real_tok / "README.md").read_text(encoding="utf-8").startswith("---")


# ---------------- 토크나이저 ----------------


def _tokenizer_corpus() -> list[str]:
    texts = [
        "대한민국은 동아시아의 한반도 남부에 있는 나라이다. 수도는 서울특별시이며, 국화는 무궁화이다.",
        "The quick brown fox jumps over the lazy dog. It's I'LL we've they're don't",
        "숫자 1234567 과 3.14159, 1,000,000원, 2026-11-04",
        "이모지 🎉🔥👍🏽 👨\u200d👩\u200d👧 漢字 かな カタカナ 中文测试",
        "  공백   과\n\n\n줄바꿈\t\t탭  \r\n 처리  ",
        " 앞 공백", "뒤 공백 ", "", " ", "\n", "\t\n \n",
        "https://github.com/concrete-sangminlee/bebut?tab=readme#전체-재현-절차",
        "def f(x):\n    return x ** 2  # 주석\n\nprint(f(3))",
        '<html><body class="a">안녕</body></html>',
        # 정규식 엔진(Python regex vs Oniguruma)마다 \s 정의가 다를 수 있는 문자들
        "edge \x1c\x1d\x1e\x1f \x85 \xa0 \u2028 \u3000 \u200b \ufeff \u00ad end",
        "e\u0301 cafe\u0301 한\u1100\u1161 ㄱㄴㄷ ㅏㅑ",
        "!!!???... ---- ==== ****", "'s 's'S 'Ll", "a'b'c", "x\r\ny\rz",
        "Ⅻ ½ ² ٣ ๓ ①②",
    ]
    rng = random.Random(0)
    pool = list("가나다라 abcXYZ 012 .,!?'\"\n\t\r ") + ["\u3000", "\xa0", "\x85", "é", "ß", "🎉", "漢", "ㅋ"]
    texts += ["".join(rng.choice(pool) for _ in range(rng.randint(1, 60))) for _ in range(200)]
    with open(os.path.join(ROOT, "evals", "heldout_ko.json"), encoding="utf-8") as f:
        texts += [q["q"] for q in json.load(f)]
    return texts


def test_tokenizer_matches_bebut(bebut_tok, hf_tok):
    mismatches = []
    for t in _tokenizer_corpus():
        ours = bebut_tok.encode(t)
        theirs = hf_tok.encode(t, add_special_tokens=False)
        back = hf_tok.decode(theirs, clean_up_tokenization_spaces=False)
        if ours != theirs or back != t:
            mismatches.append((t, ours[:10], theirs[:10]))
    assert not mismatches, mismatches[:5]


def test_special_tokens(bebut_tok, hf_tok):
    for name, i in bebut_tok.special_tokens.items():
        assert hf_tok.convert_tokens_to_ids(name) == i
    s = "<|im_start|>user\n안녕<|im_end|>\n<|im_start|>assistant\n"
    # HF는 텍스트 안의 special 문자열을 special token으로 읽는다 = bebut의 allowed_special=True
    assert hf_tok.encode(s, add_special_tokens=False) == bebut_tok.encode(s, allowed_special=True)
    assert hf_tok.encode("안녕") == bebut_tok.encode("안녕")  # bos 자동 추가 없음


def test_chat_template(bebut_tok, hf_tok):
    messages = [{"role": "user", "content": "조선을 건국한 사람은 누구야?"},
                {"role": "assistant", "content": "이성계입니다."},
                {"role": "user", "content": "몇 년에?"}]
    text = hf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expected = ("<|im_start|>user\n조선을 건국한 사람은 누구야?<|im_end|>\n"
                "<|im_start|>assistant\n이성계입니다.<|im_end|>\n"
                "<|im_start|>user\n몇 년에?<|im_end|>\n"
                "<|im_start|>assistant\n")
    assert text == expected
    ids = hf_tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list):  # transformers 버전에 따라 BatchEncoding을 돌려준다
        ids = ids["input_ids"]
    assert list(ids) == bebut_tok.encode(expected, allowed_special=True)


# ---------------- safetensors ----------------


@pytest.mark.parametrize("dtype,st_dtype", [(torch.float32, "F32"), (torch.bfloat16, "BF16")])
def test_safetensors_roundtrip(tmp_path, dtype, st_dtype):
    g = torch.Generator().manual_seed(0)
    tensors = {
        "a": torch.randn(3, 5, generator=g).to(dtype),
        "scalar": torch.tensor(1.5).to(dtype),  # shape [] — 헤더 처리 경계 사례
        "odd": torch.randn(7, generator=g).to(dtype),  # 홀수 길이 — 정렬 패딩 확인
        "big": torch.randn(64, 33, generator=g).to(dtype),
    }
    path = str(tmp_path / "t.safetensors")
    export_hf.save_safetensors(tensors, path, st_dtype)
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
    assert header_len % 8 == 0
    loaded = safetensors_torch.load_file(path)
    assert set(loaded) == set(tensors)
    for k, v in tensors.items():
        assert loaded[k].dtype == dtype and loaded[k].shape == v.shape
        assert torch.equal(loaded[k], v), k


# ---------------- CLI ----------------


def test_cli_from_checkpoint(tmp_path):
    """체크포인트 파일 → CLI → transformers 로드. configs/smoke.py(vocab 2048)를 쓴다."""
    from configs import smoke

    corpus = ["안녕하세요. 저는 학교에 갑니다.", "The quick brown fox. It's 12345."] * 20
    tok = ByteBPETokenizer.train(corpus, vocab_size=400, verbose_every=0)
    tok_path = tmp_path / "tok.json"
    tok.save(str(tok_path))
    model = _random_gpt(smoke.model)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "step": 7, "config": "smoke"}, ckpt)
    out = tmp_path / "hf"
    subprocess.run([sys.executable, os.path.join(ROOT, "export_hf.py"), "--ckpt", str(ckpt),
                    "--out", str(out), "--tokenizer", str(tok_path), "--dtype", "bf16"],
                   check=True, cwd=ROOT, capture_output=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(str(out)).eval()
    assert hf.dtype == torch.bfloat16
    conf = json.loads((out / "config.json").read_text())
    assert conf["eos_token_id"] == tok.special_tokens["<|eos|>"]  # smoke는 sft가 아니라 베이스 모드
    idx = torch.randint(0, tok.vocab_size, (1, 16))
    with torch.no_grad():
        theirs = hf(idx).logits.float()
    ours = _full_logits(model, idx)
    # bf16 반올림만큼의 차이는 허용 — 순열이 틀리면 훨씬 크게 벌어진다
    rel = (ours - theirs).abs().max() / ours.abs().max()
    assert rel < 0.05, rel
