"""DPO 검증: 데이터 형식, 토큰 정렬, 로그확률·gradient, 시작 loss = ln2, 실제 학습 e2e."""

import copy
import json
import math
import os
import re
import subprocess
import sys

import torch
import torch.nn.functional as F

from chat import encode_messages
from dpo import collate_pairs, dpo_loss, encode_pair, parse_pair, sequence_logprobs
from tests.conftest import ROOT, make_model


def test_parse_pair_formats():
    ctx = [{"role": "user", "content": "질문"}]
    assert parse_pair({"prompt": "질문", "chosen": "좋다", "rejected": "나쁘다"}) == (ctx, "좋다", "나쁘다")
    assert parse_pair({"messages": ctx, "chosen": "좋다", "rejected": "나쁘다"}) == (ctx, "좋다", "나쁘다")
    uf = {"chosen": ctx + [{"role": "assistant", "content": "좋다"}],
          "rejected": ctx + [{"role": "assistant", "content": "나쁘다"}]}
    assert parse_pair(uf) == (ctx, "좋다", "나쁘다")
    # 문맥이 다르거나, 답이 같거나, 비어 있으면 버린다
    bad = copy.deepcopy(uf)
    bad["rejected"][0] = {"role": "user", "content": "다른 질문"}  # deepcopy는 공유 dict를 공유한 채 복사한다
    assert parse_pair(bad) is None
    assert parse_pair({"prompt": "q", "chosen": "같다", "rejected": "같다"}) is None
    assert parse_pair({"prompt": "q", "chosen": "", "rejected": "x"}) is None
    assert parse_pair({"foo": 1}) is None


def test_encoding_matches_sft_format(real_tok):
    ctx = [{"role": "user", "content": "조선을 건국한 사람은?"}]
    p, c, r = encode_pair(real_tok, ctx, "이성계입니다.", "모릅니다.", 1024)
    full, mask = encode_messages(real_tok, ctx + [{"role": "assistant", "content": "이성계입니다."}],
                                 add_generation_prompt=False)
    assert p + c == full and sum(mask) == len(c)
    x, y = collate_pairs([(p, c, r)], real_tok.special_tokens["<|pad|>"])
    assert real_tok.decode(y[0][y[0] != -100].tolist()) == "이성계입니다.<|im_end|>"
    assert real_tok.decode(y[1][y[1] != -100].tolist()) == "모릅니다.<|im_end|>"
    # x는 y를 한 칸 늦게 본다
    t = (y[0] != -100).nonzero()[0].item()
    assert x[0, t + 1] == y[0, t]


def test_truncation_keeps_responses():
    from tests.conftest import CORPUS
    from tokenizer.bpe import ByteBPETokenizer

    tok = ByteBPETokenizer.train(CORPUS[:60], vocab_size=400, verbose_every=0)
    ctx = [{"role": "user", "content": "아주 긴 질문 " * 200}]
    p, c, r = encode_pair(tok, ctx, "짧은 답입니다", "나쁜 답", 64)
    assert len(p) + max(len(c), len(r)) <= 64 and len(c) >= 4
    assert encode_pair(tok, ctx, "답", "x", 8, min_resp=4) is None


def brute_force_logprobs(model, x, y):
    logits, _ = model(x, targets=y.clamp(min=0))
    lp = F.log_softmax(logits.float(), -1).gather(-1, y.clamp(min=0)[..., None])[..., 0]
    return (lp * (y != -100)).sum(-1)


def test_sequence_logprobs_and_gradients_match_brute_force():
    torch.manual_seed(0)
    model = make_model(vocab_size=1000)
    x = torch.randint(0, 1000, (4, 20))
    y = torch.randint(0, 1000, (4, 20))
    y[:, :7] = -100
    y[2, 15:] = -100
    a = sequence_logprobs(model, x, y, chunk=5)  # 청크 경계가 시퀀스를 가로지르게
    a.sum().backward()
    ga = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()
    b = brute_force_logprobs(model, x, y)
    b.sum().backward()
    assert (a - b).abs().max() < 1e-4
    for n, p in model.named_parameters():
        assert (ga[n] - p.grad).abs().max() < 1e-5, n


def test_initial_loss_is_ln2_and_loss_direction():
    lp = torch.tensor([-5.0, -3.0])
    loss, st = dpo_loss(lp, lp - 1, lp, lp - 1, beta=0.1)  # 정책 = 참조
    assert abs(loss.item() - math.log(2)) < 1e-6 and st["margin"] == 0
    better, st2 = dpo_loss(lp + 1, lp - 2, lp, lp - 1, beta=0.1)  # chosen↑ rejected↓
    assert better.item() < math.log(2) and st2["accuracy"] == 1.0 and st2["margin"] > 0


def test_dpo_e2e_learns_preference(tmp_path, tiny_tok):
    """실제 dpo.py: 시작 val loss = ln2, 학습 후 선호(chosen)가 확실히 앞선다."""
    from configs.ci_tiny import model as mcfg
    from model import GPT

    tiny_tok.save(str(tmp_path / "tok.json"))
    torch.manual_seed(0)
    torch.save({"model": GPT(mcfg).state_dict(), "step": 0, "config": "ci_tiny"}, tmp_path / "base.pt")
    with open(tmp_path / "prefs.jsonl", "w") as f:
        for i in range(160):
            f.write(json.dumps({"prompt": f"질문 {i % 9}", "chosen": "안녕하세요. 저는 학생입니다.",
                                "rejected": "The quick brown fox."}, ensure_ascii=False) + "\n")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "dpo.py"), "--base", str(tmp_path / "base.pt"),
                        "--config", "ci_tiny_dpo", "--data", str(tmp_path / "prefs.jsonl"),
                        "--tokenizer", str(tmp_path / "tok.json"), "--out-dir", str(tmp_path / "out")],
                       cwd=tmp_path, env={**os.environ, "PYTHONPATH": ROOT},
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    start = float(re.search(r"시작 val: loss ([0-9.]+)", r.stdout).group(1))
    assert abs(start - math.log(2)) < 1e-3
    final = re.findall(r"epoch \d 종료 \| val loss ([0-9.]+) \| 보상 정확도 ([0-9.]+)", r.stdout)[-1]
    assert float(final[0]) < 0.2 and float(final[1]) == 1.0
    out = tmp_path / "out" / "ci_tiny_dpo"
    assert {"epoch0.pt", "epoch2.pt", "latest.pt"} <= set(os.listdir(out))


def test_precomputed_reference_gives_exact_ln2_under_autocast():
    """참조 로그확률을 시작 모델 자신으로(같은 autocast로) 미리 계산하면 시작 loss가 정확히 ln 2다.

    예전처럼 참조 모델을 bf16 가중치로 따로 두면 정책(fp32 + autocast)과 값이 어긋났다.
    """
    from dpo import evaluate, reference_logprobs

    torch.manual_seed(0)
    model = make_model(vocab_size=1000, dim=128, n_layers=3)
    g = torch.Generator().manual_seed(1)
    pairs = [(torch.randint(0, 1000, (int(n),), generator=g).tolist(),
              torch.randint(0, 1000, (6,), generator=g).tolist(),
              torch.randint(0, 1000, (9,), generator=g).tolist())
             for n in torch.randint(3, 20, (10,), generator=g)]
    refs = reference_logprobs(model, pairs, 0, 4, "cpu", "cpu")
    assert refs.shape == (10, 2)
    v = evaluate(model, pairs, refs, 0, 4, "cpu", "cpu", beta=0.1, ls=0.0)
    assert abs(v["loss"] - math.log(2)) < 1e-6 and abs(v["margin"]) < 1e-6
