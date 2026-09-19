"""base_1b 사전학습 모델에서 시작하는 SFT 설정.

메모리: 1.2B 전체 미세조정은 가중치·기울기·AdamW 상태만 약 19GB이고,
마이크로배치 4 × 최대 1024토큰이면 사전학습(2 × 2048)과 같은 토큰 수라 26GB 안팎.
"""

from configs.base_1b import model  # 아키텍처는 동일

train = dict(
    batch_size=4,
    grad_accum=16,         # 스텝당 64개 대화
    max_lr=2e-5,           # 사전학습 최종 학습률(3e-5)보다 낮게 — 지식은 보존하고 형식만 익힌다
    min_lr=2e-6,
    warmup_steps=50,
    epochs=2,
    weight_decay=0.0,
    grad_clip=1.0,
    loss_chunk=8192,
    save_every=500,
    max_len=1024,
)
