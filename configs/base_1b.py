"""본 학습 모델 (~1.2B) — RTX 6000 Ada 48GB 단일 GPU, 25B 토큰, 약 1개월.

파라미터 내역:
  임베딩      65,536 × 2,048              = 134M  (입출력 공유)
  층당        attn 10.5M + FFN 33.8M      = 44.3M
  24층                                     = 1,063M
  합계                                     ≈ 1.2B

토큰/파라미터 = 25B / 1.2B ≈ 21 — Chinchilla 적정선.
"""

from model import ModelConfig

model = ModelConfig(
    vocab_size=65536,
    dim=2048,
    n_layers=24,
    n_heads=16,        # head_dim = 128
    n_kv_heads=4,      # GQA 4:1 — KV 캐시 1/4
    max_seq_len=2048,
)

train = dict(
    # 벤치마크 실측: 배치 2 → 6.8K tok/s / 26.1GB, 배치 4 → 10.3K tok/s / 34.7GB.
    #
    # 배치 4가 40% 빠르지만 배치 2를 쓴다. 이 GPU는 사용자의 다른 연구
    # (vLLM 추론 서버 14GB, RAG 파이프라인 15.7GB 등)와 공유되고 그 워크로드가
    # 예고 없이 뜬다. 배치 4(34.7GB)는 그때마다 OOM으로 죽어 크래시 루프에 빠진다
    # (2026-08-06 실제로 발생). 배치 2는 21GB를 남겨 공존한다.
    # GPU를 단독으로 쓸 수 있게 되면 batch_size=4 / grad_accum=128로 재개하면 된다.
    batch_size=2,
    grad_accum=256,          # 유효 배치 = 2*256*2048 ≈ 1.05M tokens
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=2000,
    max_steps=24000,         # ≈ 25.2B tokens. 계획 때 추정은 step당 154초·43일(6.8K tok/s 기준)이었고,
                             # torch.compile로 실제는 10~11K tok/s → step당 약 100초, 약 4주
    weight_decay=0.1,
    grad_clip=1.0,
    loss_chunk=8192,
    # step 하나가 2.6분이라 500 step은 21시간. 크래시 시 손실이 커서 100으로.
    eval_every=100,          # 약 4.3시간마다
    save_every=100,          # 약 4.3시간마다 (체크포인트 14GB)
    milestone_every=4000,    # 약 7일마다 스냅샷 보관
    # 학습률 decay 구간(마지막 20%)에서 고품질 데이터로 전환한다.
    # 이 구간에서 배운 건 이후 데이터에 덮어써지지 않아 사실 정확도에 크게 기여한다.
    anneal_shard_dir="data/shards_anneal",
    anneal_from_step=19200,  # max_steps의 80% = WSD decay 시작 지점
)
