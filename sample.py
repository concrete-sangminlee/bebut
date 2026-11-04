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


def load_checkpoint(path):
    """체크포인트를 CPU로 읽는다.

    weights_only=True(텐서·기본 자료형만 허용)로 먼저 시도한다 — 임의 코드를 실행할 수
    있는 pickle을 피하는 안전한 방식이다. 우리 체크포인트는 전부 여기에 해당하지만,
    예외적인 파일이면 경고 후 기존 방식으로 읽는다(직접 만든 파일만 열 것).
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        print(f"경고: {path}를 weights_only=False로 읽습니다 (신뢰하는 파일만)", file=sys.stderr)
        return torch.load(path, map_location="cpu", weights_only=False)


def load_model(ckpt_path, tokenizer_path, device, dtype=None):
    """(model, tokenizer). dtype을 안 주면 CUDA는 bf16, 그 외는 fp32.

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
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
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


def stream(model, tok, ids, args, stop_ids, out=sys.stdout):
    """생성하며 바로 출력하고, 생성된 전체 텍스트와 토큰 수를 돌려준다."""
    device = model.tok_emb.weight.device
    gen = None
    if args.seed is not None:
        gen = torch.Generator(device=device).manual_seed(args.seed)
    dec = StreamDecoder(tok)
    pieces, n = [], 0
    for t in model.generate_stream(
            torch.tensor([ids], device=device), args.max_tokens, args.temperature, args.top_p,
            top_k=args.top_k, min_p=args.min_p, repetition_penalty=args.repetition_penalty,
            stop_ids=stop_ids, generator=gen):
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
        reply, n = stream(model, tok, ids, args, stop)
        if args.verbose:
            print(f"  [{n}토큰, {n / max(1e-6, time.time() - t0):.1f} tok/s, 문맥 {len(ids)}토큰]")
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
