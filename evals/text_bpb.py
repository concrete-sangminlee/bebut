"""같은 원문에서 잰 bits-per-byte — 토크나이저가 다른 모델끼리 나란히 놓는 지표.

perplexity는 토큰당 지표라 토크나이저가 다르면 비교할 수 없다. 원문 바이트당 비트로 바꾸면
어휘 크기·분할 방식과 무관해진다. eval.bits_per_byte는 이 저장소의 토큰화된 val shard에서만
돌아서 다른 모델을 같은 글로 잴 수 없었다. 여기서는 **원문**을 받아 각 모델의 토크나이저로
자르고, 모델이 그 글 전체에 매긴 음의 로그확률(비트)을 글의 UTF-8 바이트 수로 나눈다.

    BPB = Σ −log2 P(토큰 | 앞 문맥) / 원문 바이트 수          (낮을수록 글을 잘 예측한다)

방법:
- 문서마다 따로 잰다. 맨 앞에 시작 토큰(bebut: <|eos|> — 학습 때 문서 사이에 있던 토큰, HF 모델:
  bos가 있으면 bos, 없으면 eos)을 두고 문서의 모든 토큰을 채점한다
- 문맥 창(max_len 토큰)보다 긴 문서는 stride만큼 밀며 겹쳐 읽고, 토큰마다 한 번만 채점한다.
  모든 모델에 같은 창(기본 1024토큰)을 준다
- 토크나이저가 원문을 그대로 복원하지 못하는 문서(정규화 · <unk>)는 그 모델에 유리해지므로
  뺀다. 여러 모델을 함께 재면 **모든 모델이 복원하는 문서만** 써서 같은 글로 비교한다
- 평가는 fp32 (CUDA에서도 autocast를 쓰지 않는다)

기본 글: FineWeb-2 한국어 **test** split 앞부분. 이 저장소의 사전학습은 train split만 썼으므로
bebut에게는 held-out이다. 공개 모델이 이 글을 학습했는지는 알 수 없다(웹 수집본이라 그럴 수
있다) — 공개 모델 쪽 수치는 그만큼 유리할 수 있는 기준선으로 읽는다.

사용:
  python -m evals.text_bpb --ckpt 최종=checkpoints/base_1b/latest.pt --out logs/text_bpb.json
  python -m evals.text_bpb --hf EleutherAI/polyglot-ko-1.3b Qwen/Qwen2.5-1.5B --kb 300
  python -m evals.text_bpb --ckpt ... --text-file my.txt        # 빈 줄로 문서를 나눈 직접 고른 글
  python -m evals.text_bpb --ckpt 최종=... --baseline logs/text_bpb.json   # 예전에 잰 모델과 한 표로
"""

import argparse
import hashlib
import json
import math
import os
import time

import torch
import torch.nn.functional as F


class BebutScorer:
    """이 저장소의 체크포인트. 시작 토큰은 <|eos|> (학습 데이터에서 문서와 문서 사이에 있던 토큰).
    model=None이면 토크나이저만 (원문 복원 여부 확인용)."""

    def __init__(self, model, tok, name="bebut"):
        self.model, self.tok, self.name = model, tok, name
        self.start_id = tok.special_tokens["<|eos|>"]
        if model is not None:
            model.eval()
            self.max_len = model.cfg.max_seq_len
            self.params = sum(p.numel() for p in model.parameters())

    def encode(self, text):
        return self.tok.encode(text)

    def lossless(self, text, ids):
        return self.tok.decode(ids) == text

    @torch.no_grad()
    def logits(self, ids):
        x = torch.tensor([ids], device=self.model.tok_emb.weight.device)
        return self.model.lm_head(self.model.hidden_states(x))[0].float()  # 전 위치 logits


class HFScorer:
    """HuggingFace 인과 언어모델 (AutoModelForCausalLM). transformers는 여기서만 import한다."""

    def __init__(self, repo, device="cpu", start_id=None, tokenizer_only=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.device = repo, device
        self.tok = AutoTokenizer.from_pretrained(repo)
        if start_id is None:
            start_id = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        if start_id is None:
            raise ValueError(f"{repo}: bos·eos 토큰이 없어 문서의 첫 토큰을 채점할 수 없다")
        self.start_id = start_id
        self.model = self.params = None
        self.max_len = 1 << 30
        if not tokenizer_only:
            self.model = AutoModelForCausalLM.from_pretrained(repo).to(device=device, dtype=torch.float32).eval()
            self.params = sum(p.numel() for p in self.model.parameters())
            self.max_len = getattr(self.model.config, "max_position_embeddings", None) or self.max_len

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def lossless(self, text, ids):
        return self.tok.decode(ids, clean_up_tokenization_spaces=False) == text

    @torch.no_grad()
    def logits(self, ids):
        return self.model(torch.tensor([ids], device=self.device)).logits[0].float()


def score_doc(scorer, ids, max_len, stride):
    """문서 하나의 (음의 로그확률 합[nats], 채점한 토큰 수). 모든 토큰을 정확히 한 번씩 채점한다.

    seq = [시작 토큰] + ids. 위치 t의 토큰은 그 앞 토큰들을 문맥으로 예측한다. 창은 stride씩
    밀고, 앞 창이 이미 채점한 위치는 문맥으로만 쓴다 — 두 번째 창부터 채점 토큰은 적어도
    max_len − stride개의 문맥을 갖는다.
    """
    seq = [scorer.start_id] + list(ids)
    nats, done = 0.0, 1  # done: 여기까지(미포함) 채점했다. 0번은 시작 토큰
    for begin in range(0, len(seq), stride):
        end = min(begin + max_len, len(seq))
        if end > done:
            lp = F.log_softmax(scorer.logits(seq[begin:end])[done - 1 - begin : end - 1 - begin], dim=-1)
            tgt = torch.tensor(seq[done:end], device=lp.device)
            nats -= float(lp.gather(1, tgt[:, None]).sum())
            done = end
        if end == len(seq):
            break
    return nats, len(seq) - 1


def score_text(scorer, docs, max_len=1024, stride=512, progress=None):
    """문서 목록 → {"bpb", "bytes", "tokens", "docs", "bits_per_token", "bytes_per_token"}."""
    if not 0 < stride < max_len:
        raise ValueError(f"stride는 1 이상 max_len({max_len}) 미만이어야 한다: {stride}")
    if scorer.max_len < max_len:  # 모델 문맥이 더 짧으면 창과 stride를 함께 줄인다
        max_len = scorer.max_len
        stride = min(stride, max(1, max_len // 2))
    if not 0 < stride < max_len:
        raise ValueError(f"stride는 1 이상 max_len({max_len}) 미만이어야 한다: {stride}")
    nats = n_tok = n_bytes = 0
    for k, doc in enumerate(docs):
        ids = scorer.encode(doc)
        if not ids:
            continue
        d_nats, d_tok = score_doc(scorer, ids, max_len, stride)
        nats, n_tok, n_bytes = nats + d_nats, n_tok + d_tok, n_bytes + len(doc.encode("utf-8"))
        if progress and (k + 1) % progress == 0:
            print(f"    {k + 1}/{len(docs)} 문서, 지금까지 BPB {nats / math.log(2) / n_bytes:.4f}", flush=True)
    bits = nats / math.log(2)
    return {"bpb": bits / max(1, n_bytes), "bytes": n_bytes, "tokens": n_tok, "docs": len(docs),
            "bits_per_token": bits / max(1, n_tok), "bytes_per_token": n_bytes / max(1, n_tok)}


def lossless_docs(scorers, docs):
    """모든 토크나이저가 원문 그대로 복원하는 문서만 고른다 + 모델별로 탈락한 문서 수."""
    keep, dropped = [], {s.name: 0 for s in scorers}
    for doc in docs:
        ok = True
        for s in scorers:
            if not s.lossless(doc, s.encode(doc)):
                dropped[s.name] += 1
                ok = False
        if ok:
            keep.append(doc)
    return keep, dropped


def load_docs(kb, text_file=None):
    """{영역: 문서 목록}. 기본은 FineWeb-2 한국어 test split 앞 kb킬로바이트."""
    if text_file:
        with open(text_file, encoding="utf-8") as f:
            docs = [d.strip() for d in f.read().split("\n\n") if d.strip()]
        return {text_file: docs}
    from tokenizer.compare import _stream

    return {"웹 (FineWeb-2 ko test)": _stream("HuggingFaceFW/fineweb-2", "kor_Hang", "test", int(kb * 1000))}


def merge_baselines(out, paths):
    """이번 결과(out)에 예전 결과 파일들의 모델을 영역별로 합쳐 BPB 순으로 정렬한다.

    같은 글(sha256) · 같은 창(max_len, stride)으로 잰 것만 합친다 — 글이 다르면 BPB는 나란히 놓을 수
    없다. 돌려주는 값: ({영역: [(모델, bpb, 파라미터 수, 출처)]}, [합치지 않은 이유])."""
    table = {d: [(m, r["domains"][d]["bpb"], r.get("params"), "이번")
                 for m, r in out["models"].items() if d in r["domains"]] for d in out["text"]}
    skipped = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            base = json.load(f)
        if (base.get("max_len"), base.get("stride")) != (out["max_len"], out["stride"]):
            skipped.append(f"{path}: 문맥 창이 다르다 (max_len {base.get('max_len')} · stride {base.get('stride')})")
            continue
        for d, info in out["text"].items():
            if base.get("text", {}).get(d, {}).get("sha256") != info["sha256"]:
                skipped.append(f"{path} [{d}]: 잰 글이 다르다 (sha256 불일치 — 문서가 빠졌거나 --kb가 다르다)")
                continue
            have = {m for m, *_ in table[d]}
            table[d] += [(m, r["domains"][d]["bpb"], r.get("params"), path)
                         for m, r in base["models"].items() if d in r["domains"] and m not in have]
    return {d: sorted(rows, key=lambda r: r[1]) for d, rows in table.items()}, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[], help="이 저장소의 체크포인트 — 라벨=경로")
    ap.add_argument("--hf", nargs="*", default=[], help="HuggingFace 모델 저장소 이름")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--kb", type=float, default=300, help="영역별 글의 양 (KB)")
    ap.add_argument("--text-file", default=None, help="직접 고른 글 (빈 줄로 문서를 나눈다)")
    ap.add_argument("--max-len", type=int, default=1024, help="모든 모델에 주는 문맥 창 (토큰)")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--baseline", nargs="*", default=[],
                    help="예전 결과 JSON (예: logs/text_bpb.json) — 같은 글로 잰 모델을 한 표에 합쳐 보여준다")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.ckpt and not args.hf:
        ap.error("--ckpt 또는 --hf로 모델을 하나 이상 주세요")
    for path in args.baseline:  # 몇 시간 잰 뒤에 파일이 없다고 죽지 않게 먼저 확인한다
        if not os.path.exists(path):
            ap.error(f"--baseline 파일이 없습니다: {path}")

    from sample import load_model, pick_device
    from tokenizer.bpe import ByteBPETokenizer

    device = pick_device(args.device)
    domains = load_docs(args.kb, args.text_file)

    # 1) 토크나이저만 먼저 읽어, 모든 모델이 원문을 복원하는 문서만 남긴다 (같은 글로 비교)
    checkers = [HFScorer(repo, tokenizer_only=True) for repo in args.hf]
    if args.ckpt:
        checkers.append(BebutScorer(None, ByteBPETokenizer.load(args.tokenizer)))
    text_info = {}
    for name, docs in domains.items():
        keep, dropped = lossless_docs(checkers, docs)
        text_info[name] = {"docs": len(keep), "bytes": sum(len(d.encode("utf-8")) for d in keep),
                           # 다른 날 · 다른 기계에서 잰 결과와 같은 글인지 확인하는 지문
                           "sha256": hashlib.sha256("\x00".join(keep).encode("utf-8")).hexdigest()[:16],
                           "dropped_not_lossless": {k: v for k, v in dropped.items() if v}}
        domains[name] = keep
        print(f"[{name}] 문서 {len(keep)}개 / {text_info[name]['bytes'] / 1000:.0f}KB "
              f"(복원 안 돼 뺀 문서: {text_info[name]['dropped_not_lossless'] or '없음'})", flush=True)

    # 2) 모델을 하나씩 올려 잰다
    results = {}
    jobs = [("ckpt", spec) for spec in args.ckpt] + [("hf", repo) for repo in args.hf]
    for kind, spec in jobs:
        t0 = time.time()
        if kind == "ckpt":
            label, path = spec.split("=", 1) if "=" in spec else (spec, spec)
            model, tok = load_model(path, args.tokenizer, device, torch.float32)
            scorer = BebutScorer(model, tok, label)
        else:
            scorer = HFScorer(spec, device)
        print(f"\n===== {scorer.name} ({scorer.params / 1e6:.0f}M 파라미터) =====", flush=True)
        r = {"params": scorer.params, "domains": {}}
        for name, docs in domains.items():
            r["domains"][name] = score_text(scorer, docs, args.max_len, args.stride, progress=50)
            d = r["domains"][name]
            print(f"  [{name}] BPB {d['bpb']:.4f} | 토큰당 {d['bits_per_token']:.3f}비트 · "
                  f"{d['bytes_per_token']:.2f}바이트/토큰", flush=True)
        r["seconds"] = round(time.time() - t0)
        results[scorer.name] = r
        del scorer

    out = {"max_len": args.max_len, "stride": args.stride, "dtype": "fp32", "text": text_info, "models": results}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\n저장: {args.out}")
    if args.baseline:
        table, skipped = merge_baselines(out, args.baseline)
        for d, rows in table.items():
            print(f"\n[{d}] bits-per-byte — 낮을수록 좋다")
            for name, bpb, params, src in rows:
                size = f"{params / 1e9:.2f}B" if params else "?"
                print(f"  {bpb:.4f}  {name} ({size}){'' if src == '이번' else '  ← ' + src}")
        for why in skipped:
            print(f"  합치지 않음 — {why}")


if __name__ == "__main__":
    main()
