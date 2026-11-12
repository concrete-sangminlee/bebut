"""사전학습 루프.

레시피:
- AdamW (β=(0.9, 0.95), wd=0.1 — 행렬에만), grad clip 1.0
- WSD 스케줄 (Warmup → Stable → Decay): cosine과 달리 stable 구간이 길어
  max_steps를 나중에 늘려도 이어서 학습 가능. decay는 마지막 20%에서 선형.
- bf16 autocast (+ CUDA면 TF32, torch.compile)
- gradient accumulation으로 유효 배치 확보
- 체크포인트에 model/optimizer/dataloader step까지 저장 → 완전 재개
- 주기적 val loss(매번 같은 배치) + 생성 샘플 출력, metrics.csv 로깅 (wandb는 --wandb로 선택)
- loss·gradient가 NaN/inf가 되면 그 step을 건너뛰고, 연속되면 종료해 감시 스크립트가
  마지막 체크포인트에서 다시 띄우게 한다

사용 예:
  python train.py --config debug_30m
  python train.py --config base_1b --resume checkpoints/base_1b/latest.pt
  python train.py --config base_1b --resume ... --stop-after 19200   # 그 step 저장 후 정지
"""

import argparse
import csv
import importlib
import math
import os
import shutil
import time

import torch

from data.dataloader import ShardedDataLoader
from model import GPT
from tokenizer.bpe import ByteBPETokenizer

METRIC_FIELDS = ["step", "lr", "train_loss", "val_loss", "tokens_per_sec", "grad_norm"]
LEGACY_FIELDS = METRIC_FIELDS[:5]  # 예전 train.py가 쓰던 열
MAX_BAD_STEPS = 5  # 연속 비정상 step 허용 횟수 — 넘으면 프로세스를 끝내 재개에 맡긴다


def lr_at(step, cfg):
    """WSD: warmup → stable(max_lr) → 마지막 20% 선형 decay(min_lr까지)."""
    if step < cfg["warmup_steps"]:
        return cfg["max_lr"] * (step + 1) / cfg["warmup_steps"]
    decay_start = int(cfg["max_steps"] * 0.8)
    if step < decay_start:
        return cfg["max_lr"]
    frac = (step - decay_start) / max(1, cfg["max_steps"] - decay_start)
    return cfg["max_lr"] + frac * (cfg["min_lr"] - cfg["max_lr"])


def step_if_finite(opt, loss_val, grad_norm, bad_steps, step):
    """loss와 grad norm이 유한할 때만 opt.step(). 아니면 업데이트를 건너뛰고 연속 횟수를 센다.
    MAX_BAD_STEPS번 연속이면 예외 — 가중치를 NaN으로 덮어쓴 채 '완료'로 끝나지 않게.
    돌려주는 값은 새 연속 횟수. train.py · sft.py · dpo.py가 같이 쓴다."""
    gn = float(grad_norm)
    if math.isfinite(loss_val) and math.isfinite(gn):
        opt.step()
        return 0
    bad_steps += 1
    print(f"경고: step {step} loss {loss_val} / grad norm {gn} — 업데이트 건너뜀 "
          f"({bad_steps}/{MAX_BAD_STEPS})", flush=True)
    if bad_steps >= MAX_BAD_STEPS:
        raise RuntimeError("비정상 step이 연속됐다 — 마지막 체크포인트에서 재개할 것")
    return bad_steps


VAL_WINDOWS = 40  # val loss를 잴 윈도우 수 (batch_size와 무관하게 같은 표본)


@torch.no_grad()
def estimate_val_loss(model, loader, iters, device_type, loss_chunk=None):
    """매번 **같은** val 배치 iters개로 잰다. iters=None이면 VAL_WINDOWS개 윈도우가 되게
    (batch_size를 바꿔 재개해도 epoch 모드의 val 표본이 같다 — 예전엔 20배치 고정이라 달라졌다).

    예전에는 val 로더가 호출될 때마다 다음 배치로 넘어가서, 평가마다 다른 8만 토큰을
    쟀다. 그래서 val loss가 100 step 주기로 ±0.1씩 출렁였고, 모델 변화와 표본 노이즈를
    구분할 수 없었다. 처음으로 되감아 같은 표본을 쓰면 값의 차이가 곧 모델의 차이다.
    """
    model.eval()
    loader.reset()
    if iters is None:
        iters = max(1, VAL_WINDOWS // loader.B)
    losses = []
    for _ in range(iters):
        x, y = loader.next_batch()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y, loss_chunk=loss_chunk)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def open_metrics(path, start_step=None):
    """metrics.csv를 이어 쓴다. 예전 형식(열 5개) 파일이면 그 열만 쓴다.

    예전 train.py는 start_step == 0일 때만 헤더를 써서, --resume으로 처음 만들어진
    파일엔 헤더가 없다. 첫 줄이 헤더가 아니면 예전 열 순서로 본다.

    step >= start_step인 행은 먼저 지운다. 크래시로 죽으면 마지막 체크포인트 뒤 step의 행이
    이미 파일에 남아 있고, 재개는 체크포인트 다음 step부터 다시 쓰므로 그 구간이 두 번
    들어갔다(차트가 뒤로 접혔다). 새로 시작하면(start_step=0) 예전 행을 모두 지운다.
    start_step=None이면 지우지 않는다.
    """
    fields = METRIC_FIELDS
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        header = rows[0][:1] == ["step"]
        fields = rows[0] if header else LEGACY_FIELDS
        body = rows[1:] if header else rows
        keep = body if start_step is None else [
            r for r in body if r and r[0].strip().isdigit() and int(r[0]) < start_step]
        if len(keep) != len(body):
            tmp = path + ".tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                if header:
                    w.writerow(fields)
                w.writerows(keep)
            os.replace(tmp, path)
            print(f"metrics.csv: step {start_step} 이후 행 {len(body) - len(keep)}개를 지우고 이어 씀")
        new = False
    else:
        new = True
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader()
    return f, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/ 모듈명 (예: debug_30m)")
    ap.add_argument("--shard-dir", default="data/shards")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--anneal-shard-dir", default=None,
                    help="decay 구간에서 쓸 고품질 shard 디렉토리 (config로도 지정 가능)")
    ap.add_argument("--out-dir", default="checkpoints", help="체크포인트 상위 디렉토리")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="이 step을 마치면 체크포인트를 저장하고 정상 종료 (GPU 양보·실험용)")
    ap.add_argument("--compile", action="store_true", default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    conf = importlib.import_module(f"configs.{args.config}")
    mcfg, tcfg = conf.model, conf.train

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu" if device == "cpu" else "mps"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32

    ckpt_dir = os.path.join(args.out_dir, args.config)
    os.makedirs(ckpt_dir, exist_ok=True)

    tok = ByteBPETokenizer.load(args.tokenizer)
    # 새로 시작하는 학습은 epoch 모드(모든 토큰을 한 번씩). 예전 체크포인트를 재개하면
    # 저장된 상태에 mode가 없어서 load_state_dict가 random으로 되돌린다.
    sampling = tcfg.get("sampling", "epoch")

    def make_loaders(shard_dir):
        return (
            ShardedDataLoader(shard_dir, "train", tcfg["batch_size"], mcfg.max_seq_len,
                              device=device, mode=sampling),
            ShardedDataLoader(shard_dir, "val", tcfg["batch_size"], mcfg.max_seq_len,
                              seed=123, device=device, mode=sampling),
        )

    # Annealing(=학습률 decay 구간)에서는 고품질 데이터로 갈아탄다.
    # 학습률이 높은 구간에서 배운 건 이후 데이터에 덮어써지지만, decay 구간에서
    # 배운 건 거의 그대로 남는다. 그래서 최신 모델들은 마지막 구간에 웹 데이터 대신
    # 위키·교과서 같은 지식 밀도 높은 데이터를 쓴다.
    anneal_dir = args.anneal_shard_dir or tcfg.get("anneal_shard_dir")
    anneal_from = tcfg.get("anneal_from_step")
    if anneal_dir and anneal_from is None:
        anneal_from = int(tcfg["max_steps"] * 0.8)  # WSD decay 시작 지점과 맞춘다

    # 초기 가중치를 재현 가능하게 — 예전엔 시드가 없어서 같은 설정도 실행마다 다른 모델이 나왔다
    torch.manual_seed(tcfg.get("seed", 1337))
    model = GPT(mcfg).to(device)
    print(f"모델: {model.num_params()/1e6:.1f}M 파라미터 (임베딩 제외) | device: {device}")
    opt = torch.optim.AdamW(model.param_groups(tcfg["weight_decay"]),
                            lr=tcfg["max_lr"], betas=(0.9, 0.95), fused=(device == "cuda"))

    start_step = 0
    loader_state = None
    saved_annealing = None
    if args.resume:
        # CPU로 먼저 읽는다 — 옵티마이저 상태까지 GPU에 바로 올리면 순간 메모리가 두 배가 된다
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        loader_state = ckpt["train_loader"]
        start_step = ckpt["step"] + 1
        saved_annealing = ckpt.get("annealing")  # 이 키가 생기기 전 체크포인트면 None
        del ckpt
        print(f"재개: step {start_step}부터")

    # 저장된 로더 상태는 저장 시점에 쓰던 로더의 것이다. 전환은 anneal_from step을
    # 학습하기 전에 일어나므로, 마지막으로 끝낸 step(start_step-1)이 anneal_from 이상이면
    # 그 상태는 annealing 로더 것이다.
    #
    # 예전 구현은 이 구분 없이 재개 직후 전환하면서 로더 위치를 'step 번호'로 덮어썼다.
    # 로더는 마이크로배치마다 한 칸씩(=step당 grad_accum칸) 가므로, 재개할 때마다
    # annealing 데이터 스트림이 앞쪽으로 되감겨 이미 본 배치를 다시 학습했다.
    annealing = bool(anneal_dir) and start_step > anneal_from
    if saved_annealing is not None and saved_annealing != annealing:
        # 예: annealing 중 체크포인트를 --anneal-shard-dir 없이 재개 → 고품질 데이터 로더의
        # 위치가 웹 데이터 로더에 들어가 엉뚱한 곳부터 읽는다. 조용히 넘어가지 않는다.
        raise SystemExit(f"체크포인트는 annealing={saved_annealing}로 저장됐는데 지금 설정으로는 "
                         f"annealing={annealing}입니다 — anneal_shard_dir·anneal_from_step을 확인할 것")
    train_loader, val_loader = make_loaders(anneal_dir if annealing else args.shard_dir)
    if loader_state is not None:
        train_loader.load_state_dict(loader_state)
    if annealing:
        print(f"annealing 구간 재개 — {anneal_dir}의 저장된 위치(step {train_loader.step})부터")

    use_compile = args.compile if args.compile is not None else (device == "cuda")
    if use_compile:
        model = torch.compile(model)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="bebut", name=args.config, config=tcfg)

    metrics_f, metrics = open_metrics(os.path.join(ckpt_dir, "metrics.csv"), start_step)

    tokens_per_step = tcfg["batch_size"] * tcfg["grad_accum"] * mcfg.max_seq_len
    end_step = tcfg["max_steps"] if args.stop_after is None else min(
        tcfg["max_steps"], args.stop_after + 1)
    model.train()
    t0 = t_start = time.time()
    last_log = start_step - 1  # 처리량은 마지막 로그 이후 실제로 돈 step 수로 잰다
    bad_steps = 0
    for step in range(start_step, end_step):
        if anneal_dir and not annealing and step >= anneal_from:
            train_loader, val_loader = make_loaders(anneal_dir)
            annealing = True
            print(f"=== step {step}: annealing 데이터로 전환 ({anneal_dir}) ===", flush=True)

        lr = lr_at(step, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr

        # loss는 GPU 텐서로 모은다. 마이크로배치마다 .item()을 부르면 step당
        # grad_accum번(1.2B 설정에선 256번) CPU-GPU 동기화가 걸린다.
        loss_acc = torch.zeros((), device=device)
        for micro in range(tcfg["grad_accum"]):
            x, y = train_loader.next_batch()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y, loss_chunk=tcfg.get("loss_chunk"))
            (loss / tcfg["grad_accum"]).backward()
            loss_acc += loss.detach() / tcfg["grad_accum"]
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        loss_val, gn = loss_acc.item(), grad_norm.item()

        # 한 번의 이상치로 가중치를 망치지 않는다. 데이터는 이미 소비했으니 업데이트만 건너뛴다.
        bad_steps = step_if_finite(opt, loss_val, gn, bad_steps, step)
        opt.zero_grad(set_to_none=True)

        tps = ""
        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_step * (step - last_log) / dt  # 재개 직후엔 10 step 미만일 수 있다
            last_log = step
            done = step - start_step + 1
            eta = (time.time() - t_start) / done * (end_step - step - 1) / 3600
            print(f"step {step:6d} | loss {loss_val:.4f} | lr {lr:.2e} | gnorm {gn:.2f} "
                  f"| {tps/1e3:.0f}K tok/s | 남은 시간 ~{eta:.1f}h", flush=True)
            tps = f"{tps:.0f}"

        val_loss = ""
        if step > 0 and step % tcfg["eval_every"] == 0:
            val_loss = estimate_val_loss(model, val_loader, None, device_type, tcfg.get("loss_chunk"))
            print(f"step {step:6d} | val loss {val_loss:.4f}")
            # 생성 샘플로 정성 확인
            raw = getattr(model, "_orig_mod", model)
            ids = tok.encode("대한민국의 수도는")
            if max(ids) < mcfg.vocab_size:  # 실험용 작은 vocab 모델에 큰 토크나이저를 줬을 때 방어
                sample = raw.generate(torch.tensor([ids], device=device), 48, temperature=0.8,
                                      eos_id=tok.special_tokens["<|eos|>"])
                print("샘플:", tok.decode([t for t in sample[0].tolist() if t < tok.vocab_size]))
            model.train()
            val_loss = f"{val_loss:.4f}"

        metrics.writerow({"step": step, "lr": f"{lr:.3e}", "train_loss": f"{loss_val:.4f}",
                          "val_loss": val_loss, "tokens_per_sec": tps, "grad_norm": f"{gn:.4f}"})
        if run:
            log = {"loss": loss_val, "lr": lr, "grad_norm": gn}
            if val_loss != "":
                log["val_loss"] = float(val_loss)
            run.log(log, step=step)

        last = step == tcfg["max_steps"] - 1
        # 장기 학습용 스냅샷 — 덮어쓰지 않고 따로 남겨 되돌릴 수 있게 한다.
        # 마지막 step(max_steps-1)은 ms의 배수가 아니라서 따로 챙긴다 — 최종 모델 스냅샷이 빠졌던 실수.
        # annealing 전환 직전 step도 남긴다 — annealing 효과를 비교할 기준점이다
        # (예전에는 pause_at_19200.sh로 학습을 멈추고 latest.pt를 손으로 복사했다).
        ms = tcfg.get("milestone_every")
        pre_anneal = bool(anneal_dir) and step == anneal_from - 1
        snapshot = bool(ms) and ((step > 0 and step % ms == 0) or last or pre_anneal)
        if (step > 0 and step % tcfg["save_every"] == 0) or last or step == end_step - 1 or snapshot:
            raw = getattr(model, "_orig_mod", model)
            ckpt = {
                "model": raw.state_dict(),
                "optimizer": opt.state_dict(),
                "train_loader": train_loader.state_dict(),
                "step": step,
                "config": args.config,
                "annealing": annealing,
            }
            # 원자적 저장: 임시 파일에 다 쓴 뒤 rename.
            # 그냥 덮어쓰면 저장 중 중단됐을 때 체크포인트가 깨져 학습 전체를 잃는다.
            path = os.path.join(ckpt_dir, "latest.pt")
            tmp = path + ".tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, path)
            metrics_f.flush()

            if snapshot:
                # 스냅샷도 원자적으로. 예전엔 바로 써서, 쓰는 중에 죽으면 잘린 step_N.pt가 남고
                # (재개는 N+1부터라 다시 안 만들어진다) 마일스톤 평가가 그 파일에서 매번 죽었다.
                # latest.pt를 하드링크하면 14GB를 두 번 쓰지 않는다 — latest.pt는 다음 저장 때
                # rename으로 바뀌므로 링크한 내용은 그대로 남는다. 링크가 안 되면 복사.
                snap = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
                snap_tmp = snap + ".tmp"
                if os.path.exists(snap_tmp):
                    os.remove(snap_tmp)
                try:
                    os.link(path, snap_tmp)
                except OSError:
                    shutil.copyfile(path, snap_tmp)
                os.replace(snap_tmp, snap)
                print(f"스냅샷 저장: {snap}")
            # 정지 절차(OPERATIONS.md)가 이 줄을 보고 프로세스를 끈다 — 스냅샷까지 끝난 뒤에 찍는다
            print(f"체크포인트 저장: {path} (step {step})", flush=True)

    metrics_f.close()
    if end_step < tcfg["max_steps"]:
        print(f"--stop-after {args.stop_after}: step {end_step - 1}에서 정지 (재개 가능)")
    else:
        print("학습 완료")


if __name__ == "__main__":
    main()
