"""SFT 검증: collate 동치, 길이별 묶음, 실제 sft.py e2e (에폭별 체크포인트)."""

import json
import re
import os
import subprocess
import sys

import torch

from sft import _instruction_row, collate, make_batches, padding_efficiency
from tests.conftest import ROOT


def old_collate(batch, pad_id):
    """예전 sft.py의 루프 구현 그대로."""
    L = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), L - 1), pad_id, dtype=torch.long)
    y = torch.full((len(batch), L - 1), -100, dtype=torch.long)
    for i, (ids, mask) in enumerate(batch):
        t = torch.tensor(ids)
        x[i, : len(ids) - 1] = t[:-1]
        for j in range(len(ids) - 1):
            if mask[j + 1]:
                y[i, j] = ids[j + 1]
    return x, y


def test_collate_matches_old_implementation():
    g = torch.Generator().manual_seed(0)
    batch = []
    for n in (5, 17, 9, 2):
        ids = torch.randint(0, 100, (n,), generator=g).tolist()
        mask = (torch.rand(n, generator=g) > 0.4).int().tolist()
        batch.append((ids, mask))
    for a, b in zip(collate(batch, 7), old_collate(batch, 7)):
        assert torch.equal(a, b)


def test_length_grouping_reduces_padding_and_keeps_every_example():
    import random

    rng = random.Random(0)
    data = [([0] * rng.randint(5, 500), [1]) for _ in range(2000)]
    plain = make_batches(data, 8, seed=1, group_by_length=False)
    grouped = make_batches(data, 8, seed=1, group_by_length=True)
    assert sorted(j for b in grouped for j in b) == list(range(2000))
    assert padding_efficiency(data, grouped) > padding_efficiency(data, plain) + 0.3


def test_sft_e2e_saves_epoch_checkpoints(tmp_path, tiny_tok):
    from model import GPT
    from configs.ci_tiny import model as mcfg

    tok_path = tmp_path / "tok.json"
    tiny_tok.save(str(tok_path))
    torch.manual_seed(0)
    base = tmp_path / "base.pt"
    torch.save({"model": GPT(mcfg).state_dict(), "step": 0, "config": "ci_tiny"}, base)
    with open(tmp_path / "data.jsonl", "w") as f:
        for i in range(120):
            row = ({"messages": [{"role": "user", "content": f"질문 {i % 7}"},
                                 {"role": "assistant", "content": "안녕하세요. 오늘 날씨가 정말 좋네요."}]}
                   if i % 2 else {"instruction": "인사해줘", "output": "안녕하세요. 저는 학생입니다."})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    r = subprocess.run([sys.executable, os.path.join(ROOT, "sft.py"), "--base", str(base),
                        "--config", "ci_tiny_sft", "--tokenizer", str(tok_path),
                        "--data", str(tmp_path / "data.jsonl"), "--out-dir", str(tmp_path / "out")],
                       cwd=tmp_path, env={**os.environ, "PYTHONPATH": ROOT},
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = tmp_path / "out" / "ci_tiny_sft"
    assert {"epoch0.pt", "epoch1.pt", "latest.pt"} <= set(os.listdir(out))
    vals = [float(v) for v in re.findall(r"val loss[^0-9]*([0-9.]+)", r.stdout)]
    assert len(vals) == 3  # SFT 전 + 에폭 2개
    assert vals[-1] < vals[0] - 1.0  # 반복 데이터라 확실히 내려가야 한다

    # 학습한 모델이 대화 템플릿으로 답을 생성하고 <|im_end|>로 멈춘다
    from chat import encode_prompt
    from sample import load_model

    model, _ = load_model(str(out / "latest.pt"), str(tok_path), "cpu")
    ids = encode_prompt(tiny_tok, [{"role": "user", "content": "인사해줘"}])
    end = tiny_tok.special_tokens["<|im_end|>"]
    gen = model.generate(torch.tensor([ids]), 30, temperature=0.0, eos_id=end)[0, len(ids):].tolist()
    assert end in gen and "안녕하세요" in tiny_tok.decode(gen)


def test_instruction_row_keeps_same_rows_as_old_code():
    """예전 규칙: user = instruction (+ "\n\n" + input), user와 output이 비어 있지 않으면 채택."""
    def old_rule(inst, inp, out):
        user = inst
        if inp:
            user += "\n\n" + inp
        return bool(user) and bool(out)

    for inst, inp, out in [("질문", None, "답"), ("", "입력만", "답"), ("", None, "답"),
                           ("질문", "", ""), ("질문", "입력", "답")]:
        m = _instruction_row(inst, inp, out)
        assert (m is not None) == old_rule(inst, inp, out), (inst, inp, out)
        if m:
            assert m[0]["content"] == inst + (f"\n\n{inp}" if inp else "")
