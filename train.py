"""사전학습 루프.

레시피:
- AdamW (β=(0.9, 0.95), wd=0.1 — 행렬에만), grad clip 1.0
- WSD 스케줄 (Warmup → Stable → Decay): cosine과 달리 stable 구간이 길어
  max_steps를 나중에 늘려도 이어서 학습 가능. decay는 마지막 20%에서 선형.
- bf16 autocast (+ CUDA면 TF32, torch.compile)
- gradient accumulation으로 유효 배치 확보
- 체크포인트에 model/optimizer/dataloader step까지 저장 → 완전 재개
- 주기적 val loss + 생성 샘플 출력, metrics.csv 로깅 (wandb는 --wandb로 선택)

사용 예:
  python train.py --config debug_30m
  python train.py --config base_1b --resume checkpoints/base_1b/latest.pt
"""

import argparse
import csv
import importlib
import os
import time

import torch

from data.dataloader import ShardedDataLoader
from model import GPT
from tokenizer.bpe import ByteBPETokenizer


def lr_at(step, cfg):
    """WSD: warmup → stable(max_lr) → 마지막 20% 선형 decay(min_lr까지)."""
    if step < cfg["warmup_steps"]:
        return cfg["max_lr"] * (step + 1) / cfg["warmup_steps"]
    decay_start = int(cfg["max_steps"] * 0.8)
    if step < decay_start:
        return cfg["max_lr"]
    frac = (step - decay_start) / max(1, cfg["max_steps"] - decay_start)
    return cfg["max_lr"] + frac * (cfg["min_lr"] - cfg["max_lr"])


@torch.no_grad()
def estimate_val_loss(model, loader, iters, device_type, loss_chunk=None):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = loader.next_batch()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y, loss_chunk=loss_chunk)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/ 모듈명 (예: debug_30m)")
    ap.add_argument("--shard-dir", default="data/shards")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--anneal-shard-dir", default=None,
                    help="decay 구간에서 쓸 고품질 shard 디렉토리 (config로도 지정 가능)")
    ap.add_argument("--compile", action="store_true", default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    conf = importlib.import_module(f"configs.{args.config}")
    mcfg, tcfg = conf.model, conf.train

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu" if device == "cpu" else "mps"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32

    ckpt_dir = os.path.join("checkpoints", args.config)
    os.makedirs(ckpt_dir, exist_ok=True)

    tok = ByteBPETokenizer.load(args.tokenizer)

    def make_loaders(shard_dir):
        return (
            ShardedDataLoader(shard_dir, "train", tcfg["batch_size"], mcfg.max_seq_len,
                              device=device),
            ShardedDataLoader(shard_dir, "val", tcfg["batch_size"], mcfg.max_seq_len,
                              seed=123, device=device),
        )

    # Annealing(=학습률 decay 구간)에서는 고품질 데이터로 갈아탄다.
    # 학습률이 높은 구간에서 배운 건 이후 데이터에 덮어써지지만, decay 구간에서
    # 배운 건 거의 그대로 남는다. 그래서 최신 모델들은 마지막 구간에 웹 데이터 대신
    # 위키·교과서 같은 지식 밀도 높은 데이터를 쓴다.
    anneal_dir = args.anneal_shard_dir or tcfg.get("anneal_shard_dir")
    anneal_from = tcfg.get("anneal_from_step")
    if anneal_dir and anneal_from is None:
        anneal_from = int(tcfg["max_steps"] * 0.8)  # WSD decay 시작 지점과 맞춘다

    train_loader, val_loader = make_loaders(args.shard_dir)
    annealing = False

    model = GPT(mcfg).to(device)
    print(f"모델: {model.num_params()/1e6:.1f}M 파라미터 (임베딩 제외) | device: {device}")
    opt = torch.optim.AdamW(model.param_groups(tcfg["weight_decay"]),
                            lr=tcfg["max_lr"], betas=(0.9, 0.95), fused=(device == "cuda"))

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        train_loader.load_state_dict(ckpt["train_loader"])
        start_step = ckpt["step"] + 1
        print(f"재개: step {start_step}부터")

    use_compile = args.compile if args.compile is not None else (device == "cuda")
    if use_compile:
        model = torch.compile(model)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="llm_from_scratch", name=args.config, config=tcfg)

    metrics_path = os.path.join(ckpt_dir, "metrics.csv")
    metrics_f = open(metrics_path, "a", newline="")
    metrics = csv.writer(metrics_f)
    if start_step == 0:
        metrics.writerow(["step", "lr", "train_loss", "val_loss", "tokens_per_sec"])

    tokens_per_step = tcfg["batch_size"] * tcfg["grad_accum"] * mcfg.max_seq_len
    model.train()
    t0 = time.time()
    for step in range(start_step, tcfg["max_steps"]):
        # 재개했을 때도 올바른 데이터를 쓰도록 매 step 조건을 확인한다
        if anneal_dir and not annealing and step >= anneal_from:
            train_loader, val_loader = make_loaders(anneal_dir)
            train_loader.step = step  # (seed, step) 결정론 유지 — 재개해도 같은 순서
            annealing = True
            print(f"=== step {step}: annealing 데이터로 전환 ({anneal_dir}) ===", flush=True)

        lr = lr_at(step, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr

        loss_acc = 0.0
        for micro in range(tcfg["grad_accum"]):
            x, y = train_loader.next_batch()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y, loss_chunk=tcfg.get("loss_chunk"))
            (loss / tcfg["grad_accum"]).backward()
            loss_acc += loss.item() / tcfg["grad_accum"]
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_step * (10 if step > start_step else 1) / dt
            print(f"step {step:6d} | loss {loss_acc:.4f} | lr {lr:.2e} | {tps/1e3:.0f}K tok/s")

        val_loss = ""
        if step > 0 and step % tcfg["eval_every"] == 0:
            val_loss = estimate_val_loss(model, val_loader, 20, device_type, tcfg.get("loss_chunk"))
            print(f"step {step:6d} | val loss {val_loss:.4f}")
            # 생성 샘플로 정성 확인
            raw = getattr(model, "_orig_mod", model)
            prompt = torch.tensor([tok.encode("대한민국의 수도는")], device=device)
            sample = raw.generate(prompt, 48, temperature=0.8,
                                  eos_id=tok.special_tokens["<|eos|>"])
            print("샘플:", tok.decode(sample[0].tolist()))
            model.train()

        metrics.writerow([step, f"{lr:.3e}", f"{loss_acc:.4f}", val_loss, ""])
        if run:
            log = {"loss": loss_acc, "lr": lr}
            if val_loss != "":
                log["val_loss"] = val_loss
            run.log(log, step=step)

        if step > 0 and step % tcfg["save_every"] == 0 or step == tcfg["max_steps"] - 1:
            raw = getattr(model, "_orig_mod", model)
            ckpt = {
                "model": raw.state_dict(),
                "optimizer": opt.state_dict(),
                "train_loader": train_loader.state_dict(),
                "step": step,
                "config": args.config,
            }
            # 원자적 저장: 임시 파일에 다 쓴 뒤 rename.
            # 그냥 덮어쓰면 저장 중 중단됐을 때 체크포인트가 깨져 학습 전체를 잃는다.
            path = os.path.join(ckpt_dir, "latest.pt")
            tmp = path + ".tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, path)
            metrics_f.flush()
            print(f"체크포인트 저장: {path} (step {step})")

            # 장기 학습용 스냅샷 — 덮어쓰지 않고 따로 남겨 되돌릴 수 있게 한다
            ms = tcfg.get("milestone_every")
            # 마지막 step(max_steps-1)은 ms의 배수가 아니라서 따로 챙긴다 — 최종 모델 스냅샷이 빠졌던 실수
            if ms and (step % ms == 0 or step == tcfg["max_steps"] - 1):
                snap = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
                torch.save(ckpt, snap)
                print(f"스냅샷 저장: {snap}")

    metrics_f.close()
    print("학습 완료")


if __name__ == "__main__":
    main()
