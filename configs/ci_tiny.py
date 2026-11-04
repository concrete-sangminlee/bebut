"""CI·테스트 전용 초소형 설정 — CPU에서 수 초. 학습 루프 전체(annealing 전환, 재개,
스냅샷)를 실제로 돌려 검증하는 데 쓴다 (tests/test_train_e2e.py)."""

from model import ModelConfig

model = ModelConfig(
    vocab_size=1024,
    dim=64,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=64,
)

train = dict(
    batch_size=4,
    grad_accum=2,
    max_lr=3e-3,
    min_lr=3e-4,
    warmup_steps=2,
    max_steps=12,
    weight_decay=0.1,
    grad_clip=1.0,
    loss_chunk=128,
    eval_every=4,
    save_every=4,
    milestone_every=6,
    anneal_from_step=8,
    sampling="epoch",
)
