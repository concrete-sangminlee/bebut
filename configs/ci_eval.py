"""평가 스크립트 e2e 확인용 초소형 설정 — 실제 64K 토크나이저와 같은 vocab, 문맥 2048.

학습용이 아니다. final_eval.py·sft_eval.py를 실제 벤치마크 데이터로 끝까지 돌려볼 때 쓴다.
"""

from model import ModelConfig

model = ModelConfig(vocab_size=65536, dim=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=2048)
