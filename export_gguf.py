"""bebut 체크포인트 → GGUF(llama.cpp · Ollama) 내보내기.

GGUF는 llama.cpp가 읽는 단일 파일 형식이다. 이 파일 하나로 llama.cpp(llama-cli,
llama-server), Ollama, LM Studio, llama-cpp-python에서 CPU·Apple Silicon·GPU 어디서든
돌릴 수 있고, Q8_0 양자화를 쓰면 1.2B 모델이 1.3GB 남짓이 된다.

왜 HF 내보내기를 거쳐 convert_hf_to_gguf.py를 쓰지 않고 직접 쓰나:
- 그 스크립트는 토크나이저의 pre-tokenizer 해시를 알려진 목록과 대조하는데, 우리가 직접
  학습한 토크나이저는 목록에 없어서 "BPE pre-tokenizer was not recognized"로 멈춘다.
- GGUF 형식 자체는 "헤더 + 키·값 메타데이터 + 텐서 목록 + 정렬된 텐서 바이트"로 단순하다.
  safetensors를 직접 쓴 export_hf.py와 같은 원칙(무거운 의존성 없음)으로 직접 쓴다.

구조 대응: bebut은 Qwen3와 같은 구성이라 llama.cpp의 "qwen3" 아키텍처로 내보낸다.
llama.cpp의 qwen3는 NEOX 방식 RoPE(rotate_half)를 쓰므로 HF 내보내기와 똑같은
q/k 행 재배열이 필요하다 — export_hf.convert_state_dict를 그대로 쓰고 이름만 바꾼다.
토크나이저는 "gpt2"(byte-level BPE) + pre-tokenizer "llama-bpe"로 적는다. llama-bpe의
분할 정규식은 Llama 3(=GPT-4 cl100k) 패턴으로 tokenizer/pretokenize.py와 같은 규칙이다
(tests/test_export_gguf.py가 실제 llama.cpp 토큰화와 대조해 검증한다).

사용 예:
  python export_gguf.py --ckpt checkpoints/sft_1b/latest.pt --out gguf/bebut-1.2b-sft.gguf --type q8_0
  llama-cli -m gguf/bebut-1.2b-sft.gguf -cnv              # llama.cpp 대화
  cd gguf && ollama create bebut -f Modelfile && ollama run bebut   # Ollama

--type: f32 | f16 | bf16 | q8_0 | q4_0. norm 같은 1차원 텐서는 항상 F32로 둔다(llama.cpp 관례 —
작아서 아낄 게 없고 정밀도가 중요하다). Q8_0은 32개 값마다 fp16 스케일 하나 + int8 32개(값당
8.5비트), Q4_0은 fp16 스케일 하나 + 4비트 32개(값당 4.5비트). q4_0에서도 토큰 임베딩은 Q8_0으로
둔다 — 출력층과 가중치를 공유해서 양자화 오차가 logit에 바로 드러나는 텐서다(llama.cpp도 이
텐서는 더 높은 정밀도로 남긴다). 1.2B 기준 f16 2.4GB · q8_0 1.28GB · q4_0 0.74GB.

품질: 구조가 같은 학습된 1.7B 모델에서 잰 perplexity 변화는 f16 ±0.00% · q8_0 +0.28% · q4_0 +12.9%
(docs/INFERENCE.md, evals/quant_quality.py). 기본 권장은 q8_0이고, 더 작은 파일이 필요하면 f16으로
내보낸 뒤 llama.cpp의 llama-quantize로 K-양자화(Q6_K · Q5_K_M)를 만드는 편이 q4_0보다 낫다.
"""

import argparse
import importlib
import json
import os
import struct

import numpy as np
import torch

from export_hf import CHAT_TEMPLATE, bytes_to_unicode, convert_state_dict, is_chat_config
from tokenizer.bpe import ByteBPETokenizer

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32

# GGUF 메타데이터 값 타입 (gguf.GGUFValueType)
_U32, _I32, _F32, _BOOL, _STR, _ARR = 4, 5, 6, 7, 8, 9

# ggml 텐서 타입 (gguf.GGMLQuantizationType)과 general.file_type (gguf.LlamaFileType)
GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0, GGML_BF16 = 0, 1, 2, 8, 30
QTYPES = {  # --type → (2차원 텐서의 ggml 타입, general.file_type)
    "f32": (GGML_F32, 0),
    "f16": (GGML_F16, 1),
    "bf16": (GGML_BF16, 32),
    "q8_0": (GGML_Q8_0, 7),
    "q4_0": (GGML_Q4_0, 2),
}
Q8_BLOCK = 32  # Q8_0 · Q4_0 모두 32개 값이 한 블록
TOKEN_NORMAL, TOKEN_CONTROL, TOKEN_UNUSED = 1, 3, 5


# ---------------- 메타데이터 값 ----------------


class _V:
    """타입을 명시한 메타데이터 값. 파이썬 int만 보고는 u32인지 i32인지 알 수 없어서."""

    def __init__(self, vtype, value, elem=None):
        self.vtype, self.value, self.elem = vtype, value, elem


def u32(v):
    return _V(_U32, int(v))


def f32(v):
    return _V(_F32, float(v))


def boolean(v):
    return _V(_BOOL, bool(v))


def string(v):
    return _V(_STR, str(v))


def str_array(v):
    return _V(_ARR, list(v), _STR)


def i32_array(v):
    return _V(_ARR, list(v), _I32)


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_scalar(vtype, value) -> bytes:
    if vtype == _U32:
        return struct.pack("<I", value)
    if vtype == _I32:
        return struct.pack("<i", value)
    if vtype == _F32:
        return struct.pack("<f", value)
    if vtype == _BOOL:
        return struct.pack("<?", value)
    if vtype == _STR:
        return _pack_str(value)
    raise ValueError(f"지원하지 않는 값 타입: {vtype}")


def _pack_kv(key: str, v: _V) -> bytes:
    out = _pack_str(key) + struct.pack("<I", v.vtype)
    if v.vtype != _ARR:
        return out + _pack_scalar(v.vtype, v.value)
    out += struct.pack("<IQ", v.elem, len(v.value))
    if v.elem == _I32:  # 6만 개짜리 배열은 numpy로 한 번에
        return out + np.asarray(v.value, dtype="<i4").tobytes()
    return out + b"".join(_pack_scalar(v.elem, x) for x in v.value)


# ---------------- 텐서 인코딩 ----------------


def quantize_q8_0(x: np.ndarray) -> bytes:
    """llama.cpp의 quantize_row_q8_0_ref와 같은 규칙으로 Q8_0 블록을 만든다.

    마지막 축을 32개씩 묶어 d = max|x| / 127, q = round(x / d) (0.5는 0에서 먼 쪽으로 —
    C의 roundf와 같게). 블록 = fp16 d(2바이트) + int8 q 32개 = 34바이트.
    """
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, Q8_BLOCK)
    amax = np.abs(x).max(axis=1)
    d = amax / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    v = x * inv[:, None]
    # floor(|v| + 0.5)는 float32에서 0.49999997 + 0.5가 1.0으로 반올림돼 roundf와 달랐다.
    # gguf-py와 같은 정확한 식: floor(a) + floor(2·(a − floor(a)))
    a = np.abs(v)
    f = np.floor(a)
    q = (np.sign(v) * (f + np.floor(2 * (a - f)))).astype(np.int8)
    blocks = np.empty(len(x), dtype=[("d", "<f2"), ("qs", "i1", (Q8_BLOCK,))])
    blocks["d"] = d.astype(np.float16)
    blocks["qs"] = q
    return blocks.tobytes()


def dequantize_q8_0(raw: bytes, shape) -> np.ndarray:
    """검증용 역변환 (테스트와 디버깅에서 쓴다)."""
    blocks = np.frombuffer(raw, dtype=[("d", "<f2"), ("qs", "i1", (Q8_BLOCK,))])
    return (blocks["qs"].astype(np.float32) * blocks["d"].astype(np.float32)[:, None]).reshape(shape)


def quantize_q4_0(x: np.ndarray) -> bytes:
    """llama.cpp의 quantize_row_q4_0_ref와 같은 규칙으로 Q4_0 블록을 만든다.

    32개씩 묶어, 절댓값이 가장 큰 값 m(부호 포함)으로 d = m / −8을 잡고 q = trunc(x / d + 8.5)를
    0~15로 자른다. 복원값은 d·(q − 8): m 쪽 끝은 정확히(−8칸), 반대쪽 끝은 7칸까지만 표현된다.
    블록 = fp16 d(2바이트) + 4비트 32개(앞 16개가 아래 니블, 뒤 16개가 위 니블) = 18바이트.
    """
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, Q8_BLOCK)
    m = np.take_along_axis(x, np.abs(x).argmax(axis=1)[:, None], axis=1)
    d = m / np.float32(-8)
    inv = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d != 0)
    q = np.trunc(x * inv + np.float32(8.5)).astype(np.uint8).clip(0, 15)
    blocks = np.empty(len(x), dtype=[("d", "<f2"), ("qs", "u1", (Q8_BLOCK // 2,))])
    blocks["d"] = d[:, 0].astype(np.float16)
    blocks["qs"] = q[:, : Q8_BLOCK // 2] | (q[:, Q8_BLOCK // 2 :] << np.uint8(4))
    return blocks.tobytes()


def dequantize_q4_0(raw: bytes, shape) -> np.ndarray:
    """검증용 역변환."""
    blocks = np.frombuffer(raw, dtype=[("d", "<f2"), ("qs", "u1", (Q8_BLOCK // 2,))])
    q = np.concatenate([blocks["qs"] & 0x0F, blocks["qs"] >> 4], axis=1).astype(np.int8) - 8
    return (q.astype(np.float32) * blocks["d"].astype(np.float32)[:, None]).reshape(shape)


def _ggml_type_for(t: torch.Tensor, qtype: str, name: str = "") -> int:
    if t.dim() < 2:
        return GGML_F32
    gtype = QTYPES[qtype][0]
    if gtype == GGML_Q4_0 and name == "token_embd.weight":
        gtype = GGML_Q8_0  # 출력층과 공유하는 임베딩은 한 단계 높은 정밀도로
    if gtype in (GGML_Q8_0, GGML_Q4_0) and t.shape[-1] % Q8_BLOCK:
        return GGML_F16  # 행 길이가 32의 배수가 아니면 블록을 못 만든다
    return gtype


def _nbytes(t: torch.Tensor, gtype: int) -> int:
    n = t.numel()
    return {GGML_F32: 4 * n, GGML_F16: 2 * n, GGML_BF16: 2 * n,
            GGML_Q8_0: n // Q8_BLOCK * (2 + Q8_BLOCK),
            GGML_Q4_0: n // Q8_BLOCK * (2 + Q8_BLOCK // 2)}[gtype]


QUANT_CHUNK = 1 << 22  # 한 번에 양자화하는 값의 수 (블록 크기 32의 배수)


def _quantize_chunked(t: torch.Tensor, quantize) -> bytes:
    """텐서를 QUANT_CHUNK개씩 잘라 양자화해 이어 붙인다 (블록끼리 독립이라 결과는 통째로 한 것과 같다).

    양자화는 중간 배열을 텐서 크기의 예닐곱 배 만든다. 통째로 하면 임베딩 하나(1.2B는 1.3억 값,
    vocab 15만인 모델은 3억 값)에 3~8GB가 더 들어, 15GB 기계에서 1.7B 모델을 내보내다 죽었다."""
    flat = t.reshape(-1)
    return b"".join(quantize(flat[i : i + QUANT_CHUNK].float().numpy())
                    for i in range(0, flat.numel(), QUANT_CHUNK))


def _tensor_bytes(t: torch.Tensor, gtype: int) -> bytes:
    t = t.detach().contiguous().cpu()
    if gtype == GGML_F32:
        return t.float().numpy().astype("<f4", copy=False).tobytes()
    if gtype == GGML_F16:
        return t.to(torch.float16).numpy().astype("<f2", copy=False).tobytes()
    if gtype == GGML_BF16:  # numpy엔 bf16이 없어 비트 그대로 int16으로 본다
        return t.to(torch.bfloat16).view(torch.int16).numpy().astype("<i2", copy=False).tobytes()
    if gtype == GGML_Q8_0:
        return _quantize_chunked(t, quantize_q8_0)
    if gtype == GGML_Q4_0:
        return _quantize_chunked(t, quantize_q4_0)
    raise ValueError(gtype)


# ---------------- 이름·메타데이터 ----------------

_LAYER_MAP = {
    "input_layernorm": "attn_norm",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "post_attention_layernorm": "ffn_norm",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def gguf_name(hf_name: str) -> str:
    """HF(Qwen3) 텐서 이름 → llama.cpp 이름. output.weight는 없다 (임베딩 공유)."""
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    parts = hf_name.split(".")  # model.layers.{i}.<모듈...>.weight
    if parts[:2] != ["model", "layers"] or parts[-1] != "weight":
        raise ValueError(f"대응 규칙이 없는 텐서: {hf_name}")
    module = ".".join(parts[3:-1])
    if module not in _LAYER_MAP:
        raise ValueError(f"대응 규칙이 없는 텐서: {hf_name}")
    return f"blk.{parts[2]}.{_LAYER_MAP[module]}.weight"


def tokenizer_metadata(tok: ByteBPETokenizer, vocab_size: int, chat: bool) -> dict:
    """llama.cpp의 "gpt2" BPE 토크나이저 메타데이터.

    토큰 문자열은 HF tokenizer.json과 같은 GPT-2 바이트→유니코드 표기다. llama.cpp는
    토큰 수가 임베딩 행 수와 같아야 하므로, 모델 vocab이 토크나이저보다 크면(실험용 설정)
    남는 자리를 쓰이지 않는 토큰(UNUSED)으로 채운다.
    """
    if tok.vocab_size > vocab_size:
        raise ValueError(f"토크나이저 vocab({tok.vocab_size})이 모델 vocab({vocab_size})보다 크다")
    b2u = bytes_to_unicode()
    n_bpe = 256 + len(tok.merges)
    tokens = ["".join(b2u[b] for b in tok.vocab[i]) for i in range(n_bpe)]
    types = [TOKEN_NORMAL] * n_bpe
    for text, i in sorted(tok.special_tokens.items(), key=lambda kv: kv[1]):
        assert i == len(tokens), "special token id가 연속이 아니다"
        tokens.append(text)
        types.append(TOKEN_CONTROL)
    for i in range(len(tokens), vocab_size):
        tokens.append(f"<|unused_{i}|>")
        types.append(TOKEN_UNUSED)
    if len(set(tokens)) != len(tokens):
        raise ValueError("같은 문자열의 토큰이 둘 이상이다 — 문자열 vocab으로 표현할 수 없다")
    sp = tok.special_tokens
    merges = [f"{tokens[a]} {tokens[b]}" for a, b in tok.merges]
    return {
        "tokenizer.ggml.model": string("gpt2"),
        "tokenizer.ggml.pre": string("llama-bpe"),
        "tokenizer.ggml.tokens": str_array(tokens),
        "tokenizer.ggml.token_type": i32_array(types),
        "tokenizer.ggml.merges": str_array(merges),
        "tokenizer.ggml.bos_token_id": u32(sp["<|bos|>"]),
        "tokenizer.ggml.eos_token_id": u32(sp["<|im_end|>" if chat else "<|eos|>"]),
        # 대화 턴 끝 — llama-cli/llama-server가 대화 모드에서 여기서 멈춘다
        "tokenizer.ggml.eot_token_id": u32(sp["<|im_end|>"]),
        # 문서 끝(<|eos|>)도 생성 끝으로 — 대화 모드에서 eos를 <|im_end|>로 두면 llama.cpp가 <|eos|>를
        # 생성 끝으로 보지 않아 그 뒤로 계속 생성했다(HF·Modelfile·sample.py·serve.py는 둘 다에서 멈춘다)
        "tokenizer.ggml.eom_token_id": u32(sp["<|eos|>"]),
        "tokenizer.ggml.padding_token_id": u32(sp["<|pad|>"]),
        "tokenizer.ggml.add_bos_token": boolean(False),  # 학습 때도 bos를 붙이지 않았다
        "tokenizer.ggml.add_eos_token": boolean(False),
        "tokenizer.chat_template": string(CHAT_TEMPLATE),
    }


def model_metadata(cfg, name: str, qtype: str) -> dict:
    arch = "qwen3"
    hd = cfg.dim // cfg.n_heads
    return {
        "general.architecture": string(arch),
        "general.name": string(name),
        "general.file_type": u32(QTYPES[qtype][1]),
        "general.quantization_version": u32(2),
        "general.alignment": u32(ALIGNMENT),
        f"{arch}.context_length": u32(cfg.max_seq_len),
        f"{arch}.embedding_length": u32(cfg.dim),
        f"{arch}.block_count": u32(cfg.n_layers),
        f"{arch}.feed_forward_length": u32(cfg.ffn_dim),
        f"{arch}.attention.head_count": u32(cfg.n_heads),
        f"{arch}.attention.head_count_kv": u32(cfg.n_kv_heads),
        f"{arch}.attention.key_length": u32(hd),
        f"{arch}.attention.value_length": u32(hd),
        f"{arch}.rope.freq_base": f32(cfg.rope_theta),
        f"{arch}.attention.layer_norm_rms_epsilon": f32(cfg.norm_eps),
    }


# ---------------- Ollama ----------------

# Ollama(Go 템플릿)용 ChatML. 시스템 프롬프트(Modelfile의 SYSTEM이나 요청의 system)는
# Ollama가 role "system" 메시지로 .Messages 앞에 넣어 준다.
OLLAMA_TEMPLATE = (
    "{{- range .Messages }}<|im_start|>{{ .Role }}\n"
    "{{ .Content }}<|im_end|>\n"
    "{{ end }}<|im_start|>assistant\n"
)


def modelfile(gguf_filename: str, cfg) -> str:
    return (
        f"# bebut — ollama create bebut -f Modelfile && ollama run bebut\n"
        f"FROM ./{gguf_filename}\n\n"
        f'TEMPLATE """{OLLAMA_TEMPLATE}"""\n\n'
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|eos|>"\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.95\n"
        f"PARAMETER num_ctx {cfg.max_seq_len}\n"
    )


# ---------------- 쓰기 ----------------


def write_gguf(path: str, metadata: dict, tensors: dict[str, torch.Tensor], qtype: str):
    """GGUF v3 파일을 쓴다. 텐서는 하나씩 변환해 흘려 쓴다 (수 GB를 메모리에 모으지 않는다).

    레이아웃: magic, version, 텐서 수, 키·값 수 → 키·값들 → 텐서 정보(이름, 차원 수,
    ggml 순서의 차원 = torch shape의 역순, 타입, 데이터 영역 기준 오프셋) → 정렬 패딩 →
    텐서 데이터(각각 ALIGNMENT 바이트 정렬).
    """
    infos, offset = [], 0
    for name, t in tensors.items():
        gtype = _ggml_type_for(t, qtype, name)
        infos.append((name, t, gtype, offset))
        offset += _nbytes(t, gtype)
        offset += -offset % ALIGNMENT

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata))
    for k, v in metadata.items():
        header += _pack_kv(k, v)
    for name, t, gtype, off in infos:
        dims = list(reversed(t.shape))  # ggml의 ne[0]은 가장 안쪽(연속) 차원
        header += _pack_str(name) + struct.pack("<I", len(dims))
        header += struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", gtype, off)
    header += b"\x00" * (-len(header) % ALIGNMENT)

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header)
        written = 0
        for name, t, gtype, off in infos:
            assert written == off, (name, written, off)
            data = _tensor_bytes(t, gtype)
            f.write(data)
            written += len(data)
            pad = -written % ALIGNMENT
            f.write(b"\x00" * pad)
            written += pad
    os.replace(tmp, path)  # 원자적 저장 — 중간에 끊겨도 깨진 파일이 남지 않는다


def export_gguf(state_dict: dict[str, torch.Tensor], model_cfg, tokenizer: ByteBPETokenizer,
                out_path: str, qtype: str = "f16", chat: bool = True, name: str = "bebut",
                write_modelfile: bool = True) -> str:
    """state_dict를 GGUF 한 파일로 내보낸다. 같은 디렉토리에 Ollama Modelfile도 쓴다."""
    if qtype not in QTYPES:
        raise ValueError(f"type은 {list(QTYPES)} 중 하나: {qtype}")
    hf = convert_state_dict(state_dict, model_cfg)  # RoPE 재배열 + 이름 정리 (HF 내보내기와 공유)
    tensors = {gguf_name(k): v for k, v in hf.items()}
    metadata = {**model_metadata(model_cfg, name, qtype),
                **tokenizer_metadata(tokenizer, model_cfg.vocab_size, chat)}
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    write_gguf(out_path, metadata, tensors, qtype)
    if write_modelfile:
        with open(os.path.join(out_dir, "Modelfile"), "w", encoding="utf-8") as f:
            f.write(modelfile(os.path.basename(out_path), model_cfg))
    return out_path


def main():
    ap = argparse.ArgumentParser(description="bebut 체크포인트를 GGUF(llama.cpp·Ollama)로 내보낸다")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="출력 .gguf 경로 (같은 폴더에 Modelfile도 생긴다)")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--type", choices=list(QTYPES), default="q8_0")
    ap.add_argument("--name", default=None, help="general.name (기본: 체크포인트 config 이름)")
    ap.add_argument("--chat", action=argparse.BooleanOptionalAction, default=None,
                    help="eos를 <|im_end|>로 둘지 (기본: SFT·DPO 체크포인트면 켬)")
    args = ap.parse_args()

    from sample import load_checkpoint  # CPU로 읽고 가능하면 weights_only=True

    ckpt = load_checkpoint(args.ckpt)
    conf = importlib.import_module(f"configs.{ckpt['config']}")
    chat = args.chat if args.chat is not None else is_chat_config(str(ckpt["config"]))
    tok = ByteBPETokenizer.load(args.tokenizer)
    name = args.name or f"bebut-{ckpt['config']}"
    export_gguf(ckpt["model"], conf.model, tok, args.out, args.type, chat, name)
    size = os.path.getsize(args.out)
    print(f"내보내기 완료: {args.out} ({args.type}, {size / 1e9:.2f}GB, {'대화' if chat else '베이스'} 모드)")
    print(json.dumps({"llama.cpp": f"llama-cli -m {args.out} -cnv",
                      "ollama": f"cd {os.path.dirname(args.out) or '.'} && ollama create bebut -f Modelfile"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
