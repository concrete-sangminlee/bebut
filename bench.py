"""합성 데이터로 학습 처리량 측정 — 실제 학습 전 소요 시간 추정용.

사용: python bench.py --config base_1b [--compile]
"""

import argparse
import importlib
import time

import torch

from model import GPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    conf = importlib.import_module(f"configs.{args.config}")
    mcfg, tcfg = conf.model, conf.train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    model = GPT(mcfg).to(device)
    print(f"{args.config}: {model.num_params()/1e6:.1f}M 파라미터 (임베딩 제외), "
          f"전체 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.param_groups(0.1), lr=1e-4, fused=(device == "cuda"))

    B, T = tcfg["batch_size"], mcfg.max_seq_len
    x = torch.randint(0, mcfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, mcfg.vocab_size, (B, T), device=device)

    model.train()
    for i in range(args.steps):
        if i == 2:  # 워밍업/컴파일 제외
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x, y, loss_chunk=tcfg.get("loss_chunk"))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0

    n = args.steps - 2
    micro_tok = B * T
    tps = micro_tok * n / dt
    step_tok = micro_tok * tcfg["grad_accum"]
    print(f"마이크로배치 {B}x{T} | {tps/1e3:.1f}K tok/s")
    print(f"최적화 step당 {step_tok/1e6:.2f}M tokens, {step_tok/tps:.1f}초")
    print(f"설정된 {tcfg['max_steps']} steps = {step_tok*tcfg['max_steps']/1e9:.1f}B tokens, "
          f"{step_tok*tcfg['max_steps']/tps/3600:.1f}시간 ({step_tok*tcfg['max_steps']/tps/86400:.1f}일)")
    print(f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()
