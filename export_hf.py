"""bebut 체크포인트 → HuggingFace(Qwen3) 형식 내보내기.

왜 Qwen3인가: bebut의 구성(pre-RMSNorm, RoPE, GQA, RoPE 이전 head 단위 QK-RMSNorm,
SwiGLU, bias 없음, 입출력 임베딩 공유)은 Qwen3ForCausalLM과 정확히 같다. 그래서 새
아키텍처 코드를 쓰지 않고도 transformers · vLLM · llama.cpp(convert_hf_to_gguf.py)가
바로 읽는다.

차이는 두 가지뿐이고 둘 다 가중치 재배열로 해결된다:
1) RoPE 차원 짝짓기 — bebut은 인접한 (x[2j], x[2j+1])을 회전하고(GPT-J/원조 Llama 방식),
   HF는 (x[j], x[j+hd/2])를 회전한다(rotate_half). head마다 q/k projection의 출력
   행을 [0,2,4,…,1,3,5,…] 순서로 바꾸면 같은 연산이 된다. q·k 내적은 양쪽에 같은
   순열을 적용하면 변하지 않는다. QK-Norm 가중치도 RoPE 전에 원소별로 곱해지므로
   같은 순열로 옮긴다 (RMS 자체는 순열에 불변).
2) 토크나이저 표현 — bebut은 merge를 (id, id) 쌍으로 저장하지만 HF `tokenizers`는
   GPT-2의 bytes→unicode 문자열 vocab과 "A B" merge 문자열을 쓴다. id는 그대로 둔다
   (바이트 b → id b, merge i → 256+i, special은 그 뒤).

의존성 원칙(무거운 라이브러리 금지)에 따라 transformers·safetensors·tokenizers 없이
파일 형식을 직접 쓴다. safetensors는 "8바이트 헤더 길이 + JSON 헤더 + 원시 텐서 바이트"
라는 단순한 형식이라 직접 써도 수십 줄이다.

사용 예:
  python export_hf.py --ckpt checkpoints/sft_1b/latest.pt --out hf/bebut-1.2b-sft
  python export_hf.py --ckpt checkpoints/base_1b/latest.pt --out hf/bebut-1.2b-base --dtype fp32
"""

import argparse
import importlib
import json
import os
import struct

import torch

from sample import load_checkpoint
from tokenizer.bpe import ByteBPETokenizer
from tokenizer.pretokenize import GPT4_SPLIT_PATTERN

REPO_URL = "https://github.com/concrete-sangminlee/bebut"

DTYPES = {
    "bf16": (torch.bfloat16, "BF16", "bfloat16"),
    "fp32": (torch.float32, "F32", "float32"),
}

# SFT 학습·sample.py --chat과 정확히 같은 형식:
#   <|im_start|>{role}\n{content}<|im_end|>\n ... <|im_start|>assistant\n
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- endif -%}"
)


# ---------------- safetensors 직접 쓰기 ----------------


def _tensor_bytes(t: torch.Tensor, st_dtype: str) -> bytes:
    """텐서를 little-endian 원시 바이트로. bf16은 numpy가 모르므로 int16으로 재해석한다."""
    t = t.detach().contiguous().cpu()
    if st_dtype == "BF16":
        return t.view(torch.int16).numpy().astype("<i2", copy=False).tobytes()
    if st_dtype == "F32":
        return t.numpy().astype("<f4", copy=False).tobytes()
    raise ValueError(f"지원하지 않는 dtype: {st_dtype}")


def save_safetensors(tensors: dict[str, torch.Tensor], path: str, st_dtype: str,
                     metadata: dict[str, str] | None = None):
    """safetensors 형식으로 저장한다.

    형식: [u64 LE 헤더 길이][JSON 헤더][텐서 바이트들]. 헤더에는 이름별 dtype·shape와
    데이터 구간(data_offsets, 데이터 영역 시작 기준)이 들어간다. 헤더는 공백으로
    8바이트 정렬을 맞춘다 (텐서 데이터가 정렬된 주소에서 시작하도록).
    1.2B 모델은 수 GB라 전체를 메모리에 이어붙이지 않고 텐서별로 흘려 쓴다.
    """
    itemsize = {"BF16": 2, "F32": 4}[st_dtype]
    header: dict = {"__metadata__": {"format": "pt", **(metadata or {})}}
    offset = 0
    for name, t in tensors.items():
        n = t.numel() * itemsize
        header[name] = {"dtype": st_dtype, "shape": list(t.shape), "data_offsets": [offset, offset + n]}
        offset += n
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * (-len(hb) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for t in tensors.values():
            f.write(_tensor_bytes(t, st_dtype))
    os.replace(tmp, path)  # 원자적 저장 — 중간에 끊겨도 깨진 파일이 남지 않는다


# ---------------- 가중치 변환 ----------------


def rope_permutation(head_dim: int) -> torch.Tensor:
    """bebut(인접 쌍) → HF(rotate_half) 차원 순서. 새 j번째 = 원래 perm[j]번째."""
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


def _permute_rows(w: torch.Tensor, n_heads: int, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    """(n_heads*head_dim, in) projection의 출력 행을 head마다 perm 순서로 재배열."""
    out_dim, in_dim = w.shape
    assert out_dim == n_heads * head_dim, (w.shape, n_heads, head_dim)
    return w.view(n_heads, head_dim, in_dim)[:, perm, :].reshape(out_dim, in_dim)


def convert_state_dict(sd: dict[str, torch.Tensor], cfg) -> dict[str, torch.Tensor]:
    """bebut state_dict → Qwen3 키·레이아웃. 모르는 키가 있으면 조용히 버리지 않고 실패한다
    (아키텍처가 바뀌었는데 일부 가중치가 빠진 채 내보내지는 사고를 막기 위해)."""
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}  # torch.compile 접두사
    hd = cfg.dim // cfg.n_heads
    perm = rope_permutation(hd)
    out: dict[str, torch.Tensor] = {}
    used: set[str] = set()

    def take(k):
        used.add(k)
        return sd[k]

    out["model.embed_tokens.weight"] = take("tok_emb.weight")
    # lm_head는 tok_emb와 같은 텐서(tying) — tie_word_embeddings=true로 표현하고 한 번만 쓴다
    if "lm_head.weight" in sd:
        used.add("lm_head.weight")
        if not torch.equal(sd["lm_head.weight"], sd["tok_emb.weight"]):
            raise ValueError("lm_head와 tok_emb가 다르다 — tying 전제가 깨졌다")

    for i in range(cfg.n_layers):
        b, h = f"blocks.{i}.", f"model.layers.{i}."
        out[h + "input_layernorm.weight"] = take(b + "attn_norm.weight")
        out[h + "self_attn.q_proj.weight"] = _permute_rows(take(b + "attn.wq.weight"), cfg.n_heads, hd, perm)
        out[h + "self_attn.k_proj.weight"] = _permute_rows(take(b + "attn.wk.weight"), cfg.n_kv_heads, hd, perm)
        out[h + "self_attn.v_proj.weight"] = take(b + "attn.wv.weight")
        out[h + "self_attn.o_proj.weight"] = take(b + "attn.wo.weight")
        out[h + "self_attn.q_norm.weight"] = take(b + "attn.q_norm.weight")[perm]
        out[h + "self_attn.k_norm.weight"] = take(b + "attn.k_norm.weight")[perm]
        out[h + "post_attention_layernorm.weight"] = take(b + "ffn_norm.weight")
        out[h + "mlp.gate_proj.weight"] = take(b + "ffn.w_gate.weight")
        out[h + "mlp.up_proj.weight"] = take(b + "ffn.w_up.weight")
        out[h + "mlp.down_proj.weight"] = take(b + "ffn.w_down.weight")
    out["model.norm.weight"] = take("final_norm.weight")

    unknown = set(sd) - used
    if unknown:
        raise ValueError(f"변환 규칙이 없는 가중치: {sorted(unknown)[:10]}")
    return out


def hf_config(cfg, tok: ByteBPETokenizer | None, dtype_name: str, chat: bool) -> dict:
    sp = tok.special_tokens if tok else {}
    eos = sp.get("<|im_end|>" if chat else "<|eos|>")
    rope_theta = float(cfg.rope_theta)
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.dim,
        "intermediate_size": cfg.ffn_dim,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.dim // cfg.n_heads,
        "max_position_embeddings": cfg.max_seq_len,
        # 구버전 transformers·vLLM·llama.cpp는 rope_theta를, transformers v5는
        # rope_parameters를 읽는다. 둘 다 적어 어느 쪽에서도 같은 값을 쓰게 한다.
        "rope_theta": rope_theta,
        "rope_parameters": {"rope_type": "default", "rope_theta": rope_theta},
        "rms_norm_eps": cfg.norm_eps,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "max_window_layers": cfg.n_layers,
        "initializer_range": 0.02,
        "use_cache": True,
        "torch_dtype": dtype_name,
        "bos_token_id": sp.get("<|bos|>"),
        "eos_token_id": eos,
        "pad_token_id": sp.get("<|pad|>"),
    }


# ---------------- 토크나이저 변환 ----------------


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2의 바이트→유니코드 대응표. byte-level BPE vocab을 '보이는 문자열'로 쓰기 위한 것.

    출력 가능한 문자(! ~ ~, ¡ ~ ¬, ® ~ ÿ)는 자기 자신에, 나머지 68개 바이트(공백·제어문자 등)는
    U+0100부터 차례로 대응시킨다. 공백(0x20)이 'Ġ'가 되는 이유다.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def hf_tokenizer_json(tok: ByteBPETokenizer) -> dict:
    """`tokenizers` 라이브러리의 tokenizer.json 구조를 직접 만든다.

    - pre_tokenizer: bebut과 같은 GPT-4 정규식으로 chunk를 자른 뒤(Split, Isolated)
      ByteLevel로 바이트→문자 매핑만 한다(use_regex=false — 정규식은 이미 적용했다).
    - model: BPE. merge 순위가 곧 적용 순서라 bebut의 rank 기반 merge와 결과가 같다.
      ignore_merges=false — true면 vocab에 통째로 있는 단어를 merge 없이 바로 쓰는데,
      드물게 merge 경로와 다른 분할이 나올 수 있어 끈다.
    """
    b2u = bytes_to_unicode()

    def s(i: int) -> str:
        return "".join(b2u[b] for b in tok.vocab[i])

    vocab = {s(i): i for i in range(256 + len(tok.merges))}
    if len(vocab) != 256 + len(tok.merges):
        raise ValueError("vocab에 같은 바이트열을 가진 토큰이 있어 문자열 vocab으로 표현할 수 없다")
    merges = [f"{s(a)} {s(b)}" for a, b in tok.merges]
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True}
             for t, i in sorted(tok.special_tokens.items(), key=lambda kv: kv[1])]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": GPT4_SPLIT_PATTERN},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                 "use_regex": False},
            ],
        },
        "post_processor": None,  # bos 자동 추가 없음 — 학습 때도 붙이지 않았다
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                    "use_regex": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


def hf_tokenizer_config(tok: ByteBPETokenizer, max_len: int, chat: bool) -> dict:
    eos = "<|im_end|>" if chat else "<|eos|>"
    return {
        # Qwen2Tokenizer 등 전용 클래스는 자체 정규식으로 pre-tokenizer를 다시 만들 수 있어
        # tokenizer.json을 그대로 쓰는 범용 클래스를 지정한다.
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": eos,
        "pad_token": "<|pad|>",
        "unk_token": None,
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "model_max_length": max_len,
        "chat_template": CHAT_TEMPLATE,
        "added_tokens_decoder": {
            str(i): {"content": t, "lstrip": False, "normalized": False, "rstrip": False,
                     "single_word": False, "special": True}
            for t, i in tok.special_tokens.items()
        },
    }


def model_card(cfg, n_params: int, dtype_name: str, chat: bool, source: str | None) -> str:
    kind = "SFT 대화 모델" if chat else "사전학습 베이스 모델"
    src = f"\n원본 체크포인트: `{source}`\n" if source else ""
    return f"""---
language:
- ko
license: mit
library_name: transformers
pipeline_tag: text-generation
---

# bebut ({kind})

토크나이저부터 직접 만들어 GPU 한 장으로 학습한 한국어 decoder-only 언어 모델입니다.
학습 코드·데이터 파이프라인·평가는 [{REPO_URL}]({REPO_URL})에 있습니다.
{src}
| | |
|---|---|
| 파라미터 | {n_params / 1e9:.2f}B (임베딩 포함) |
| 구조 | dim {cfg.dim} / {cfg.n_layers}층 / GQA {cfg.n_heads}:{cfg.n_kv_heads} / ctx {cfg.max_seq_len} |
| 형식 | Qwen3ForCausalLM 호환 (bebut 원본에서 가중치 재배열만 함) |
| dtype | {dtype_name} |
| 토크나이저 | byte-level BPE {cfg.vocab_size:,} vocab (한국어 코퍼스로 직접 학습) |

## 사용법

### transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("<이 디렉토리>")
model = AutoModelForCausalLM.from_pretrained("<이 디렉토리>", torch_dtype="auto")

messages = [{{"role": "user", "content": "조선을 건국한 사람은 누구야?"}}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
out = model.generate(ids, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.95)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

### vLLM

```bash
vllm serve <이 디렉토리> --max-model-len {cfg.max_seq_len}
```

### llama.cpp (GGUF)

```bash
python convert_hf_to_gguf.py <이 디렉토리> --outfile bebut.gguf
```

`convert_hf_to_gguf.py`가 "BPE pre-tokenizer was not recognized" 오류를 내면, 출력된
`chkhsh`를 `get_vocab_base_pre()`에 `res = "llama-bpe"`로 추가하세요. 이 토크나이저의
분할 정규식은 GPT-4(cl100k) 패턴으로 llama-bpe와 같은 계열입니다.

## 주의

- 채팅 형식은 `<|im_start|>{{role}}\\n{{content}}<|im_end|>\\n`입니다(내장 chat_template).
  SFT는 user/assistant 단일 턴으로만 학습했습니다.
- 생성은 `<|im_end|>`(대화) 또는 `<|eos|>`(문서 끝)에서 멈춥니다(generation_config.json).
- 입력은 NFC로 정규화해서 넣으세요(`unicodedata.normalize("NFC", text)`). 원래 저장소의 대화
  경로는 메시지를 NFC로 바꾸지만 이 토크나이저는 바이트 단위 그대로라, 맥·PDF에서 온 NFD 한글은
  5배 넘는 바이트 조각 토큰이 됩니다(학습 때 본 적 없는 형태).
- assistant 기록이 공백·줄바꿈으로 시작하면, 저장소(SFT)는 `assistant\n`과 내용을 따로 토큰화하지만
  chat_template은 이어서 토큰화해 경계 토큰이 달라질 수 있습니다(보통의 대화에서는 같습니다).
- HF 토크나이저는 입력 텍스트 안의 `<|im_start|>` 같은 문자열을 special token으로 해석합니다.
  사용자 입력을 그대로 넣을 때는 `split_special_tokens=True`를 고려하세요.
- 계산을 못 하고, 드문 사실은 그럴듯하게 지어냅니다. 한계와 평가는 저장소의
  README·docs/RESULTS.md를 보세요.
"""


# ---------------- 내보내기 ----------------


def is_chat_config(config_name: str) -> bool:
    """SFT·DPO를 거친 체크포인트면 대화 모델(eos = <|im_end|>)로 본다.

    설정 이름의 "_" 조각 중 하나가 sft나 dpo로 시작하면 대화 모델 — sft_1b, dpo_1b,
    ci_tiny_sft, ci_tiny_dpo 모두 해당한다 (startswith만 보면 ci_tiny_dpo를 놓친다).
    """
    return any(part.startswith(("sft", "dpo")) for part in str(config_name).split("_"))


def export(state_dict: dict[str, torch.Tensor], model_cfg, tokenizer: ByteBPETokenizer | None,
           out_dir: str, dtype: str = "bf16", chat: bool = True, source: str | None = None):
    """state_dict를 HF 디렉토리로 내보낸다. tokenizer=None이면 가중치·config만 쓴다."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype은 {list(DTYPES)} 중 하나: {dtype}")
    torch_dtype, st_dtype, dtype_name = DTYPES[dtype]
    if tokenizer is not None and tokenizer.vocab_size > model_cfg.vocab_size:
        raise ValueError(f"토크나이저 vocab({tokenizer.vocab_size})이 모델 vocab({model_cfg.vocab_size})보다 크다")
    os.makedirs(out_dir, exist_ok=True)

    tensors = {k: v.to(torch_dtype) for k, v in convert_state_dict(state_dict, model_cfg).items()}
    save_safetensors(tensors, os.path.join(out_dir, "model.safetensors"), st_dtype)
    n_params = sum(t.numel() for t in tensors.values())

    def dump(name, obj):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    dump("config.json", hf_config(model_cfg, tokenizer, dtype_name, chat))
    if tokenizer is not None:
        sp = tokenizer.special_tokens
        dump("generation_config.json", {
            "bos_token_id": sp["<|bos|>"],
            # 대화 턴 끝과 문서 끝 둘 다에서 멈춘다
            "eos_token_id": [sp["<|im_end|>"], sp["<|eos|>"]],
            "pad_token_id": sp["<|pad|>"],
            # sample.py 기본값과 같다
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
        })
        with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(hf_tokenizer_json(tokenizer), f, ensure_ascii=False)
        dump("tokenizer_config.json", hf_tokenizer_config(tokenizer, model_cfg.max_seq_len, chat))
        dump("special_tokens_map.json", {
            "bos_token": "<|bos|>",
            "eos_token": "<|im_end|>" if chat else "<|eos|>",
            "pad_token": "<|pad|>",
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        })
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card(model_cfg, n_params, dtype_name, chat, source))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="bebut 체크포인트를 HuggingFace(Qwen3) 형식으로 내보낸다")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="출력 디렉토리")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument("--chat", action=argparse.BooleanOptionalAction, default=None,
                    help="eos를 <|im_end|>로 둘지 (기본: SFT·DPO 체크포인트면 켬)")
    args = ap.parse_args()

    # mmap으로 읽는다 — 사전학습 체크포인트엔 옵티마이저 상태(1.2B 기준 약 10GB)도 들어 있다.
    # 예전엔 그것까지 전부 메모리에 올렸다(15GB 기계에서 OOM). export_gguf.py와 같은 함수
    ckpt = load_checkpoint(args.ckpt)
    conf = importlib.import_module(f"configs.{ckpt['config']}")
    chat = args.chat if args.chat is not None else is_chat_config(ckpt["config"])
    tok = ByteBPETokenizer.load(args.tokenizer)
    export(ckpt["model"], conf.model, tok, args.out, args.dtype, chat,
           source=f"{args.ckpt} (config {ckpt['config']}, step {ckpt.get('step')})")
    print(f"내보내기 완료: {args.out} ({args.dtype}, {'대화' if chat else '베이스'} 모드)")
    print("확인: python -c \"from transformers import AutoModelForCausalLM; "
          f"AutoModelForCausalLM.from_pretrained('{args.out}')\"")


if __name__ == "__main__":
    main()
