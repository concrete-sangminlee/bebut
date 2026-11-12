"""체크포인트에서 텍스트 생성. --chat이면 SFT 모델용 대화 모드.

생성은 토큰이 나오는 대로 바로 출력한다(스트리밍). 대화 기록은 컨텍스트 창
(max_seq_len)에 맞게 오래된 턴부터 잘라 넣는다.

사용 예:
  python sample.py --ckpt checkpoints/base_1b/latest.pt --prompt "대한민국의 수도는"
  python sample.py --ckpt checkpoints/sft_1b/latest.pt --chat
  python sample.py --ckpt checkpoints/sft_1b/latest.pt --chat --repetition-penalty 1.1

대화 중 명령: /reset (기록 지우기), /system <내용> (시스템 프롬프트), /exit
"""

import argparse
import importlib
import pickle
import sys
import time

import torch

from chat import StreamDecoder, fit_messages
from model import GPT
from tokenizer.bpe import ByteBPETokenizer


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def device_type_of(device: str) -> str:
    """'cuda:1' → 'cuda' (autocast·fused AdamW용). 예전 sft.py·dpo.py는 문자열이 정확히 'cuda'일
    때만 CUDA로 봐서 --device cuda:0이면 autocast 대상이 'mps'가 되어 bf16 없이 fp32로 돌았다."""
    return torch.device(device).type


def load_checkpoint(path):
    """체크포인트를 CPU로 읽는다.

    weights_only=True(텐서·기본 자료형만 허용)로 먼저 시도한다 — 임의 코드를 실행할 수
    있는 pickle을 피하는 안전한 방식이다. 우리 체크포인트는 전부 여기에 해당하지만,
    예외적인 파일이면 경고 후 기존 방식으로 읽는다(직접 만든 파일만 열 것).

    mmap=True로 연다: 사전학습 체크포인트엔 옵티마이저 상태(1.2B 기준 약 10GB)가 들어 있는데,
    예전엔 그것까지 전부 메모리에 올라갔다(14GB 체크포인트 + 모델 사본 → 약 19GB, 15GB 기계에서
    추론 불가). mmap이면 load_state_dict가 실제로 읽는 가중치만 페이지로 올라온다.
    zip 형식이 아닌 아주 오래된 파일은 mmap을 못 써서 예전 방식으로 읽는다.
    """
    def load(weights_only):
        try:
            return torch.load(path, map_location="cpu", weights_only=weights_only, mmap=True)
        except RuntimeError as e:
            if "mmap" not in str(e):
                raise
            return torch.load(path, map_location="cpu", weights_only=weights_only)

    try:
        return load(True)
    except pickle.UnpicklingError:
        print(f"경고: {path}를 weights_only=False로 읽습니다 (신뢰하는 파일만)", file=sys.stderr)
        return load(False)


def cpu_has_fast_bf16() -> bool:
    """CPU가 bf16 행렬곱을 하드웨어로 지원하는지 (AVX512-BF16 / AMX).

    지원하면 bf16이 fp32보다 빠르다 — 디코드는 매 토큰 가중치 전체를 읽는 메모리 대역폭
    싸움이라 가중치가 절반 크기인 bf16이 유리하다 (1.2B 실측 fp32 8.2 → bf16 12.1 tok/s,
    docs/INFERENCE.md). 지원하지 않는 CPU에서 bf16은 소프트웨어로 흉내 내 오히려 느리다.
    """
    # mkldnn의 _is_mkldnn_bf16_supported()는 AVX-512만 있어도(bf16 명령 없이 흉내) True라서
    # Skylake·Ice Lake 같은 CPU에서 bf16을 고르게 된다 — 거기선 prefill이 fp32보다 3배 넘게 느리다.
    # 실제 bf16 명령(AVX512-BF16) 또는 AMX 타일이 있는지를 직접 본다.
    try:
        avx512_bf16 = getattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)()
        amx = getattr(torch.cpu, "_is_amx_tile_supported", lambda: False)()
    except RuntimeError:
        return False
    return bool(avx512_bf16 or amx)


def default_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda") or (device == "cpu" and cpu_has_fast_bf16()):
        return torch.bfloat16
    return torch.float32


def load_model(ckpt_path, tokenizer_path, device, dtype=None):
    """(model, tokenizer). dtype을 안 주면 CUDA·bf16 지원 CPU는 bf16, 그 외는 fp32.
    평가 스크립트는 점수가 dtype에 흔들리지 않게 fp32를 명시해서 부른다.

    CPU로 먼저 읽는 이유: 사전학습 체크포인트엔 옵티마이저 상태(1.2B 기준 약 10GB)가
    들어 있다. 예전처럼 map_location=device로 읽으면 쓰지도 않을 그 10GB가 GPU에 올라갔다.
    """
    ckpt = load_checkpoint(ckpt_path)
    conf = importlib.import_module(f"configs.{ckpt['config']}")
    model = GPT(conf.model)
    model.load_state_dict(ckpt["model"])
    model.meta = {k: ckpt[k] for k in ("step", "config") if k in ckpt}
    del ckpt
    if dtype is None:
        dtype = default_dtype(device)
    model = model.to(device=device, dtype=dtype).eval()
    tok = ByteBPETokenizer.load(tokenizer_path)
    return model, tok


def add_sampling_args(ap):
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="1.1 안팎이면 반복 루프가 줄어든다 (1.0 = 끔)")
    ap.add_argument("--seed", type=int, default=None)


def stream(model, tok, ids, args, stop_ids, out=sys.stdout, cache=None):
    """생성하며 바로 출력하고, 생성된 전체 텍스트와 토큰 수를 돌려준다.

    cache(model.new_cache())를 넘기면 지난 호출과 겹치는 앞부분의 K·V를 재사용한다.
    """
    device = model.tok_emb.weight.device
    gen = None
    if args.seed is not None:
        gen = torch.Generator(device=device).manual_seed(args.seed)
    dec = StreamDecoder(tok)
    pieces, n = [], 0
    for t in model.generate_stream(
            torch.tensor([ids], device=device), args.max_tokens, args.temperature, args.top_p,
            top_k=args.top_k, min_p=args.min_p, repetition_penalty=args.repetition_penalty,
            stop_ids=stop_ids, generator=gen, cache=cache):
        n += 1
        piece = dec.push(int(t))
        pieces.append(piece)
        out.write(piece)
        out.flush()
    tail = dec.flush()
    out.write(tail + "\n")
    return "".join(pieces) + tail, n


def chat_loop(model, tok, args):
    stop = [tok.special_tokens["<|im_end|>"], tok.special_tokens["<|eos|>"]]
    # 답변 자리를 남기고 남는 만큼만 기록을 넣는다 (max_tokens가 너무 크면 절반은 문맥에 양보)
    budget = max(model.cfg.max_seq_len // 2, model.cfg.max_seq_len - args.max_tokens)
    messages = [{"role": "system", "content": args.system}] if args.system else []
    # 턴을 넘겨 K·V를 재사용한다 — 매 턴 기록 전체를 다시 계산하지 않고 새 부분만 prefill
    cache = None if args.no_prefix_cache else model.new_cache()
    print("대화 모드 — 빈 입력 또는 /exit 종료, /reset 기록 삭제, /system <내용> 시스템 프롬프트")
    while True:
        try:
            user = input("\n사용자: ").strip()
        except EOFError:
            break
        if not user or user == "/exit":
            break
        if user == "/reset":
            messages = [m for m in messages if m["role"] == "system"]
            print("(기록을 지웠습니다)")
            continue
        if user.startswith("/system"):
            messages = [m for m in messages if m["role"] != "system"]
            content = user.removeprefix("/system").strip()
            if content:
                messages.insert(0, {"role": "system", "content": content})
            print(f"(시스템 프롬프트: {content or '없음'})")
            continue
        messages.append({"role": "user", "content": user})
        ids = fit_messages(tok, messages, budget)
        print("모델: ", end="", flush=True)
        t0 = time.time()
        reply, n = stream(model, tok, ids, args, stop, cache=cache)
        if args.verbose:
            reused = cache.reused if cache is not None else 0
            print(f"  [{n}토큰, {n / max(1e-6, time.time() - t0):.1f} tok/s, 문맥 {len(ids)}토큰 "
                  f"중 {reused}토큰 캐시 재사용]")
        messages.append({"role": "assistant", "content": reply.strip()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--prompt", default="대한민국의 수도는")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--system", default=None, help="시스템 프롬프트 (SFT 데이터엔 없던 형식)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["auto", "fp32", "bf16"], default="auto")
    ap.add_argument("--verbose", action="store_true", help="토큰 수·속도 표시")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="턴 사이 KV 캐시 재사용 끄기 (bf16에서 완전한 재현이 필요할 때)")
    add_sampling_args(ap)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = {"auto": None, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    model, tok = load_model(args.ckpt, args.tokenizer, device, dtype)

    if args.chat:
        chat_loop(model, tok, args)
    else:
        ids = tok.encode(args.prompt)[-(model.cfg.max_seq_len - 1):]
        print(args.prompt, end="", flush=True)
        stream(model, tok, ids, args, [tok.special_tokens["<|eos|>"]])


if __name__ == "__main__":
    main()
