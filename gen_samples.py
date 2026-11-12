"""학습 중인 체크포인트에서 CPU로 샘플 여러 개 생성.

GPU는 학습이 쓰고 있으므로 CUDA를 아예 안 보이게 하고 CPU로만 돌린다.
체크포인트를 한 번만 로드해 여러 프롬프트를 처리한다.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU 접근 차단 — 학습을 방해하지 않기 위해

import time

import torch

torch.set_num_threads(32)  # 학습의 데이터 로딩용 코어를 남겨둔다

from model import GPT
from sample import load_checkpoint
from tokenizer.bpe import ByteBPETokenizer
import configs.base_1b as conf

CKPT = "checkpoints/base_1b/latest.pt"

PROMPTS = [
    ("대한민국의 수도는", 0.8),   # 학습 중 쓰는 프롬프트 (비교용)
    ("대한민국의 수도는", 0.8),   # 같은 프롬프트 반복 — 무작위성 확인
    ("인공지능이란", 0.7),
    ("서울대학교는", 0.7),
    ("김치를 담그는 방법은", 0.7),
    ("조선은 1392년에", 0.7),
    ("지구 온난화의 가장 큰 원인은", 0.6),
]

t0 = time.time()
ckpt = load_checkpoint(CKPT)
model = GPT(conf.model)
model.load_state_dict(ckpt["model"])
model.eval()
tok = ByteBPETokenizer.load("tokenizer/tokenizer.json")
print(f"체크포인트 step {ckpt['step']} 로드 ({time.time()-t0:.0f}초)\n", flush=True)

eos = tok.special_tokens["<|eos|>"]
for i, (prompt, temp) in enumerate(PROMPTS, 1):
    t = time.time()
    ids = tok.encode(prompt)
    out = model.generate(torch.tensor([ids]), max_new_tokens=100,
                         temperature=temp, top_p=0.95, eos_id=eos)
    text = tok.decode(out[0].tolist()).replace("<|eos|>", " [문서 끝]")
    n = out.shape[1] - len(ids)
    print(f"[{i}] temperature={temp}, {n}토큰, {time.time()-t:.0f}초")
    print(text)
    print(flush=True)
