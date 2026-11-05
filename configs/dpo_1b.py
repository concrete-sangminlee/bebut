"""sft_1b 결과에서 시작하는 DPO 설정.

메모리: 정책 모델 학습(가중치·기울기·AdamW) 약 19GB + 배치 2쌍(= 4시퀀스 × 최대
1024토큰) 활성값. SFT(4 × 1024)와 같은 토큰 수라 26GB 안팎. 참조 모델은 메모리에 두지
않고 학습 전에 참조 로그확률을 한 번 계산해 둔다 (데이터 전체에 forward 한 번).
학습률은 SFT(2e-5)보다 훨씬 낮게 — DPO는 조금만 커도 응답이 길어지거나 망가진다.
"""

from configs.base_1b import model  # noqa: F401 — 아키텍처는 동일

train = dict(
    batch_size=2,          # 마이크로배치 = 선호 쌍 2개 (chosen+rejected 4시퀀스)
    grad_accum=32,         # step당 64쌍
    max_lr=5e-7,
    warmup_frac=0.1,
    epochs=1,
    beta=0.1,              # 클수록 참조(SFT) 모델에서 덜 벗어난다
    label_smoothing=0.0,
    weight_decay=0.0,
    grad_clip=1.0,
    max_len=1024,
    save_every=200,
    log_every=10,
)
