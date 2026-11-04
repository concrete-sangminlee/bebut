"""pytest 공용 설정 — 저장소 루트를 import 경로에 넣고, 작은 모델·토크나이저를 공유한다."""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import GPT, ModelConfig  # noqa: E402
from tokenizer.bpe import ByteBPETokenizer  # noqa: E402

# 작은 BPE 학습용 말뭉치 — 한국어·영어·숫자·특수문자를 고루 섞는다
CORPUS = [
    "안녕하세요. 저는 서울대학교 학생입니다.",
    "안녕하세요. 오늘 날씨가 정말 좋네요.",
    "저는 오늘 학교에 갑니다. 학교에서 공부합니다.",
    "대한민국의 수도는 서울입니다. 비가 왔다. 그래서 땅이 젖었다.",
    "질문: 이것은 사실인가?\n답: 예\n답: 아니오",
    "The quick brown fox jumps over the lazy dog. It's a test 12345.",
] * 40


@pytest.fixture(scope="session")
def tiny_tok():
    return ByteBPETokenizer.train(CORPUS, vocab_size=700, verbose_every=0)


@pytest.fixture(scope="session")
def real_tok():
    return ByteBPETokenizer.load(os.path.join(ROOT, "tokenizer", "tokenizer.json"))


def make_model(vocab_size=1000, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=vocab_size, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
               max_seq_len=128)
    cfg.update(kw)
    return GPT(ModelConfig(**cfg)).eval()


@pytest.fixture
def tiny_model():
    return make_model()
