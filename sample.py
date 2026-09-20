"""체크포인트에서 텍스트 생성. --chat이면 SFT 모델용 대화 모드.

사용 예:
  python sample.py --ckpt checkpoints/base_1b/latest.pt --prompt "대한민국의 수도는"
  python sample.py --ckpt checkpoints/sft_1b/latest.pt --chat
"""

import argparse
import importlib

import torch

from model import GPT
from tokenizer.bpe import ByteBPETokenizer


def load_model(ckpt_path, tokenizer_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    conf = importlib.import_module(f"configs.{ckpt['config']}")
    model = GPT(conf.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = ByteBPETokenizer.load(tokenizer_path)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--prompt", default="대한민국의 수도는")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, tok = load_model(args.ckpt, args.tokenizer, device)
    eos = tok.special_tokens["<|eos|>"]

    if args.chat:
        # SFT 학습과 동일한 chat template
        print("대화 모드 (빈 입력으로 종료)")
        history = ""
        while True:
            user = input("\n사용자: ").strip()
            if not user:
                break
            history += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            ids = tok.encode(history, allowed_special=True)
            out = model.generate(torch.tensor([ids], device=device), args.max_tokens,
                                 args.temperature, args.top_p,
                                 eos_id=tok.special_tokens["<|im_end|>"])
            reply = tok.decode(out[0, len(ids):].tolist()).removesuffix("<|im_end|>").strip()
            print(f"모델: {reply}")
            history += f"{reply}<|im_end|>\n"
    else:
        ids = tok.encode(args.prompt)
        out = model.generate(torch.tensor([ids], device=device), args.max_tokens,
                             args.temperature, args.top_p, eos_id=eos)
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
