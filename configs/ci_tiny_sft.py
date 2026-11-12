"""CI·테스트 전용 SFT 설정 — configs/ci_tiny.py 모델과 짝 (tests/test_sft.py::test_sft_e2e_saves_epoch_checkpoints)."""

from configs.ci_tiny import model  # noqa: F401 — 아키텍처 재사용 (train.py가 conf.model을 읽는다)

train = dict(
    batch_size=4,
    grad_accum=2,
    max_lr=3e-3,
    min_lr=3e-4,
    warmup_steps=2,
    epochs=2,
    weight_decay=0.0,
    grad_clip=1.0,
    loss_chunk=128,
    save_every=100,
    max_len=64,
)
