"""학습 곡선과 벤치마크 추이 그래프 생성 (README용).

라벨은 영어로 쓴다 — matplotlib 기본 폰트에 한글 글리프가 없어 네모로 깨진다."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOK_PER_STEP = 2 * 256 * 2048  # batch * grad_accum * seq_len

steps, loss, vstep, vloss = [], [], [], []
with open("logs/metrics_base_1b.csv") as f:
    for r in csv.DictReader(f):
        if not r["train_loss"]:
            continue
        steps.append(int(r["step"])); loss.append(float(r["train_loss"]))
        if r["val_loss"]:
            vstep.append(int(r["step"])); vloss.append(float(r["val_loss"]))

ms = list(csv.DictReader(open("logs/milestones_base_1b.csv")))
mtok = [int(m["tokens"]) / 1e9 for m in ms]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# 1) 학습 곡선 — 이동평균으로 노이즈 완화
W = 50
sm = [sum(loss[max(0, i - W):i + 1]) / len(loss[max(0, i - W):i + 1]) for i in range(len(loss))]
tok = [s * TOK_PER_STEP / 1e9 for s in steps]
ax[0].plot(tok, loss, lw=0.4, alpha=0.25, color="tab:blue")
ax[0].plot(tok, sm, lw=1.6, color="tab:blue", label="train (moving avg 50)")
ax[0].plot([s * TOK_PER_STEP / 1e9 for s in vstep], vloss, lw=0.9, color="tab:orange",
           alpha=0.8, label="val")
ax[0].axvline(19200 * TOK_PER_STEP / 1e9, color="gray", ls="--", lw=1)
ax[0].text(19200 * TOK_PER_STEP / 1e9, max(loss) * 0.92, " annealing\n switch", fontsize=8, color="gray")
ax[0].set_xlabel("tokens (B)"); ax[0].set_ylabel("loss"); ax[0].set_title("1.2B pretraining loss")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

# 2) perplexity
ax[1].plot(mtok, [float(m["ppl"]) for m in ms], "o-", color="tab:red")
ax[1].axvline(19200 * TOK_PER_STEP / 1e9, color="gray", ls="--", lw=1)
ax[1].set_xlabel("tokens (B)"); ax[1].set_ylabel("perplexity")
ax[1].set_title("held-out perplexity"); ax[1].grid(alpha=0.25)

# 3) KOBEST
for key, lab, ch in (("copa", "COPA", 50), ("hellaswag", "HellaSwag", 25)):
    ax[2].plot(mtok, [float(m[key]) for m in ms], "o-", label=lab)
    ax[2].axhline(ch, ls=":", lw=1, color="gray")
ax[2].axvline(19200 * TOK_PER_STEP / 1e9, color="gray", ls="--", lw=1)
ax[2].set_xlabel("tokens (B)"); ax[2].set_ylabel("accuracy (%)")
ax[2].set_title("KOBEST (dotted = random)"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("docs/training_curves.png", dpi=130)
print("저장: docs/training_curves.png")
