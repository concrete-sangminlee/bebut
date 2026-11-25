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

    with open(tmp_path / "extra.jsonl", "w") as f:  # --extra-data는 학습에만 더해진다
        for _ in range(16):
            f.write(json.dumps({"instruction": "12 더하기 7은?", "output": "19입니다."},
                               ensure_ascii=False) + "\n")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "sft.py"), "--base", str(base),
                        "--config", "ci_tiny_sft", "--tokenizer", str(tok_path),
                        "--data", str(tmp_path / "data.jsonl"), "--out-dir", str(tmp_path / "out"),
                        "--extra-data", str(tmp_path / "extra.jsonl")],
                       cwd=tmp_path, env={**os.environ, "PYTHONPATH": ROOT},
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "추가 데이터" in r.stdout and "train 112 / val 24" in r.stdout  # val은 기본 데이터만
    out = tmp_path / "out" / "ci_tiny_sft"
    assert {"epoch0.pt", "epoch1.pt", "latest.pt"} <= set(os.listdir(out))
    vals = [float(v) for v in re.findall(r"val loss[^0-9]*([0-9.]+)", r.stdout)]
    assert len(vals) == 3  # SFT 전 + 에폭 2개
    assert vals[-1] < vals[0] - 1.0  # 반복 데이터라 확실히 내려가야 한다

    # 학습한 모델이 대화 템플릿으로 답을 생성하고 <|im_end|>로 멈춘다
    from chat import encode_prompt
    from sample import load_model

    model, _ = load_model(str(out / "latest.pt"), str(tok_path), "cpu", torch.float32)  # 기계와 무관하게
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


def _run_sft(tmp, out, *extra, ok=True):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "sft.py"), "--config", "ci_tiny_sft",
                        "--tokenizer", str(tmp / "tok.json"), "--data", str(tmp / "data.jsonl"),
                        "--out-dir", str(tmp / out), *extra],
                       cwd=tmp, env={**os.environ, "PYTHONPATH": ROOT},
                       capture_output=True, text=True, timeout=300)
    assert (r.returncode == 0) == ok, r.stdout[-2000:] + r.stderr[-2000:]
    return r


def test_sft_resume_is_bit_exact(tmp_path, tiny_tok):
    """끊고 재개한 SFT의 가중치가 중단 없이 학습한 것과 비트 단위로 같다.

    에폭 중간(step 5)과 에폭 마지막 step(13 — val·epoch0.pt 저장 직전)에서 끊는다. 예전엔 재개
    수단이 없어서 latest.pt를 --base로 줄 수밖에 없었고, 그러면 학습률 스케줄과 AdamW 모멘트가
    초기화돼 다른 모델이 됐다."""
    from configs.ci_tiny import model as mcfg
    from model import GPT

    tiny_tok.save(str(tmp_path / "tok.json"))
    torch.manual_seed(0)
    base = str(tmp_path / "base.pt")
    torch.save({"model": GPT(mcfg).state_dict(), "step": 0, "config": "ci_tiny"}, base)
    with open(tmp_path / "data.jsonl", "w") as f:
        for i in range(140):  # train 112 → 에폭당 14 step, 2 에폭
            f.write(json.dumps({"instruction": f"질문 {i % 11}", "output": f"답은 {i % 5}번입니다. 안녕하세요."},
                               ensure_ascii=False) + "\n")

    _run_sft(tmp_path, "full", "--base", base)
    resume = str(tmp_path / "split" / "ci_tiny_sft" / "resume.pt")
    r = _run_sft(tmp_path, "split", "--base", base, "--stop-after", "5")
    assert "step 5에서 정지" in r.stdout and os.path.exists(resume)
    r = _run_sft(tmp_path, "split", "--resume", resume, "--stop-after", "13")
    assert "재개" in r.stdout and "epoch 0, step 6부터" in r.stdout
    assert not os.path.exists(tmp_path / "split" / "ci_tiny_sft" / "epoch0.pt")  # 에폭 끝 처리 전에 끊겼다
    r = _run_sft(tmp_path, "split", "--resume", resume)
    assert "SFT 완료" in r.stdout and not os.path.exists(resume)  # 끝나면 재개 상태는 지운다

    for name in ("epoch0.pt", "epoch1.pt", "latest.pt"):
        a = torch.load(tmp_path / "full" / "ci_tiny_sft" / name)
        b = torch.load(tmp_path / "split" / "ci_tiny_sft" / name)
        assert a["step"] == b["step"]
        for k in a["model"]:
            assert torch.equal(a["model"][k], b["model"][k]), (name, k)
    # 끊긴 실행과 이어 한 실행의 val loss가 중단 없는 실행의 에폭별 val loss와 같다
    vals = re.findall(r"epoch (\d) 종료 \| val loss ([0-9.]+)", r.stdout)
    full = re.findall(r"epoch (\d) 종료 \| val loss ([0-9.]+)",
                      _run_sft(tmp_path, "full2", "--base", base).stdout)
    assert vals == full and len(vals) == 2

    # 데이터가 바뀌면 재개를 거절한다 (위치가 다른 예제를 가리키게 된다)
    _run_sft(tmp_path, "other", "--base", base, "--stop-after", "2")
    with open(tmp_path / "data.jsonl", "a") as f:
        f.write(json.dumps({"instruction": "새 질문", "output": "새 답입니다. 안녕하세요."}, ensure_ascii=False) + "\n")
    r = _run_sft(tmp_path, "other", "--resume", str(tmp_path / "other" / "ci_tiny_sft" / "resume.pt"), ok=False)
    assert "데이터나 설정이 저장할 때와 다릅니다" in r.stdout + r.stderr
    r = _run_sft(tmp_path, "none", ok=False)  # --base도 --resume도 없으면
    assert "--base" in r.stderr
