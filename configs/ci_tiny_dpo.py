"""CI·테스트 전용 DPO 설정 — configs/ci_tiny.py 모델과 짝 (tests/test_dpo.py)."""

from configs.ci_tiny import model  # noqa: F401 — 아키텍처 재사용

train = dict(
    batch_size=4,
    grad_accum=2,
    max_lr=3e-3,
    warmup_frac=0.1,
    epochs=3,
    beta=0.5,
    weight_decay=0.0,
    grad_clip=1.0,
    max_len=64,
    save_every=100,
    log_every=5,
)
