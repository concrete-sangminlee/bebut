"""양자화 품질 — 학습된 모델에서 F16 · Q8_0 · Q4_0이 perplexity를 얼마나 바꾸는가.

docs/INFERENCE.md는 CPU에서 GGUF Q8_0을 권하지만, 그 품질은 무작위 가중치의 logit 오차로만
확인했었다. 무작위 가중치는 근거가 약하다 — 학습된 가중치에는 양자화 오차를 키우는 큰 값(이상치)이
있다. 이 환경에는 학습된 bebut 체크포인트가 없으므로, **구조가 같은(Qwen3) 학습된 공개 모델**을
이 저장소의 GPT로 읽어(export_hf.import_state_dict) 이 저장소의 양자화기로 GGUF를 만들고,
llama.cpp로 같은 글의 perplexity를 잰다. bebut 1.2B와 구조·크기가 비슷한 모델(기본 Qwen3-1.7B-Base)의
결과이지 bebut 자체의 결과는 아니다 — 체크포인트가 있으면 --ckpt로 직접 잰다.

재는 것 (같은 토큰 묶음으로):
  기준      이 저장소의 PyTorch 경로, fp32
  f16 등    llama.cpp로 읽은 GGUF (KV 캐시는 fp32). ppl, 기준 대비 변화율, 토큰별 손실 차이의 평균
            절댓값, 1순위 예측이 기준과 같은 비율

--llama-types를 주면, 이 저장소가 구현하지 않은 형식(K-양자화 등)을 llama.cpp 자체 양자화기로
F16 파일에서 만들어 같은 방법으로 잰다 — "4비트가 필요하면 무엇을 쓸까"의 근거로.

--torch-check를 주면 교차 확인을 한다: 양자화했다가 되돌린 가중치를 PyTorch 모델에 넣어 같은
묶음을 잰다. llama.cpp로 읽은 GGUF의 수치와 비슷하면 그 손실은 양자화 자체의 것이고, 크게 다르면
내보내기나 추론 경로가 틀린 것이다 (결과의 torch_dequant · vs_torch_dequant).

llama-cpp-python · transformers · safetensors가 필요하다 (저장소 의존성이 아니라 이 측정에만 쓴다).

사용:
  python -m evals.quant_quality --llama-types q4_k_m q5_k_m q6_k --out logs/quant_quality.json  # Qwen3-1.7B-Base
  python -m evals.quant_quality --ckpt checkpoints/sft_1b/latest.pt --out ... # bebut 체크포인트
"""

import argparse
import ctypes
import gc
import importlib
import json
import math
import os
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from export_gguf import (GGML_Q4_0, GGML_Q8_0, QTYPES, QUANT_CHUNK, _ggml_type_for, dequantize_q4_0,
                         dequantize_q8_0, export_gguf, quantize_q4_0, quantize_q8_0)
from export_hf import import_state_dict, model_config_from_hf
from model import GPT
from tokenizer.bpe import ByteBPETokenizer


def chunks_of(ids: list[int], ctx: int, limit: int | None = None) -> list[list[int]]:
    """토큰열을 ctx개씩 겹치지 않게 자른다 (모자란 꼬리는 버린다)."""
    out = [ids[i : i + ctx] for i in range(0, len(ids) - ctx + 1, ctx)]
    return out[:limit] if limit else out


def nll_and_top1(logits: np.ndarray, ids: list[int]):
    """한 묶음의 logits (T, V) → (토큰별 음의 로그확률 T−1개, 1순위 예측 T−1개). 첫 토큰은 문맥으로만."""
    lp = F.log_softmax(torch.from_numpy(np.ascontiguousarray(logits[:-1], dtype=np.float32)), dim=-1)
    tgt = torch.tensor(ids[1:])
    return -lp.gather(1, tgt[:, None])[:, 0].numpy(), lp.argmax(-1).numpy()


def summarize(nll: np.ndarray, top1: np.ndarray, ref_nll=None, ref_top1=None) -> dict:
    r = {"ppl": float(math.exp(nll.mean())), "nll": float(nll.mean()), "tokens": int(nll.size)}
    if ref_nll is not None:
        r["ppl_change_pct"] = 100 * (r["ppl"] / math.exp(ref_nll.mean()) - 1)
        r["mean_abs_nll_diff"] = float(np.abs(nll - ref_nll).mean())
        r["top1_agreement"] = float((top1 == ref_top1).mean())
    return r


@torch.no_grad()
def torch_reference(model, chunks):
    nll, top1 = [], []
    for ids in chunks:
        logits = model.lm_head(model.hidden_states(torch.tensor([ids])))[0].float().numpy()
        a, b = nll_and_top1(logits, ids)
        nll.append(a)
        top1.append(b)
    return np.concatenate(nll), np.concatenate(top1)


@torch.no_grad()
def torch_dequant_scores(model, chunks, types):
    """교차 확인용: 타입마다 가중치를 '양자화 → 역양자화'한 값으로 바꿔 PyTorch로 잰다 → {타입: (nll, top1)}.

    텐서별 형식은 GGUF에 쓸 때와 같은 함수(_ggml_type_for)로 정한다 (1차원 텐서는 그대로, Q4_0의
    임베딩은 Q8_0). 끝나면 가중치를 원래대로 돌려놓는다. 메모리는 모델 + 원본 사본만큼 든다."""
    codecs = {GGML_Q8_0: (quantize_q8_0, dequantize_q8_0), GGML_Q4_0: (quantize_q4_0, dequantize_q4_0)}
    params = dict(model.named_parameters())
    # 원본 보관: 공개 모델의 가중치는 bf16으로 저장돼 있어 bf16 사본이 정확하다(메모리 절반). 아니면 그대로 복사
    orig = {k: (v.to(torch.bfloat16) if torch.equal(v.to(torch.bfloat16).float(), v) else v.clone())
            for k, v in params.items()}
    out = {}
    for q in types:
        for name, p in params.items():
            gname = "token_embd.weight" if name in ("tok_emb.weight", "lm_head.weight") else name
            g = _ggml_type_for(p, q, gname)
            if g not in codecs:
                p.copy_(orig[name])
                continue
            quantize, dequantize = codecs[g]
            dst, src = p.view(-1), orig[name].reshape(-1)
            for i in range(0, src.numel(), QUANT_CHUNK):  # 조각 단위로 — 통째로 하면 임베딩 하나에 수 GB가 든다
                w = src[i : i + QUANT_CHUNK].float()
                dst[i : i + w.numel()] = torch.from_numpy(dequantize(quantize(w.numpy()), (w.numel(),)))
        out[q] = torch_reference(model, chunks)
    for name, p in params.items():
        p.copy_(orig[name])
    return out


def llama_scores(path, chunks, ctx, threads):
    from llama_cpp import Llama

    llm = Llama(model_path=path, n_ctx=ctx, logits_all=True, verbose=False, n_threads=threads,
                type_k=0, type_v=0)  # KV 캐시는 fp32 — 가중치 양자화의 영향만 본다
    nll, top1 = [], []
    for ids in chunks:
        llm.reset()
        llm.eval(ids)
        a, b = nll_and_top1(np.array(llm.scores[: len(ids)]), ids)
        nll.append(a)
        top1.append(b)
    del llm
    return np.concatenate(nll), np.concatenate(top1)


def llama_quantize(src, dst, ftype, threads):
    """llama.cpp 자체 양자화기로 GGUF(src, F16)를 ftype(예: "q4_k_m")으로 다시 양자화해 dst에 쓴다."""
    import llama_cpp

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, f"LLAMA_FTYPE_MOSTLY_{ftype.upper()}")
    params.nthread = threads
    if llama_cpp.llama_model_quantize(src.encode(), dst.encode(), ctypes.byref(params)) != 0:
        raise RuntimeError(f"llama.cpp 양자화 실패: {ftype}")
    return dst


def load_hf_model(repo, max_seq_len):
    """HF Qwen3 모델 → (이 저장소의 GPT[fp32], 토큰화 함수)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as f:
        cfg = model_config_from_hf(json.load(f), max_seq_len=max_seq_len)
    sd = import_state_dict(load_file(hf_hub_download(repo, "model.safetensors")), cfg)
    model = GPT(cfg)
    target = model.state_dict()
    if set(target) != set(sd):
        raise ValueError(f"가중치 이름이 맞지 않는다: {sorted(set(target) ^ set(sd))[:6]}")
    with torch.no_grad():  # 텐서마다 제자리에 복사 (bf16 → fp32). 전체를 한 번 더 복사하지 않는다
        for k, v in sd.items():
            target[k].copy_(v)
    del sd, target
    tok = AutoTokenizer.from_pretrained(repo)
    return model.eval(), lambda text: tok.encode(text, add_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="Qwen/Qwen3-1.7B-Base", help="구조가 같은(Qwen3) 학습된 HF 모델")
    ap.add_argument("--ckpt", default=None, help="bebut 체크포인트 (주면 --hf 대신 이것을 잰다)")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--types", nargs="+", choices=[t for t in QTYPES if t != "f32"], default=["f16", "q8_0", "q4_0"])
    ap.add_argument("--llama-types", nargs="*", default=[],
                    help="llama.cpp 양자화기로 F16에서 만들어 함께 잴 형식 (예: q4_k_m q5_k_m q6_k) — 결과에 "
                         "'llama:<형식>'으로 남는다")
    ap.add_argument("--torch-check", action="store_true",
                    help="교차 확인: 양자화→역양자화한 가중치를 PyTorch에 넣어 잰 값을 함께 남긴다 (q8_0 · q4_0)")
    ap.add_argument("--kb", type=float, default=60, help="글의 양 (KB, FineWeb-2 ko test split 앞부분)")
    ap.add_argument("--ctx", type=int, default=256, help="묶음 하나의 토큰 수")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tokenizer.compare import _stream

    t0 = time.time()
    bebut_tok = ByteBPETokenizer.load(args.tokenizer)
    if args.ckpt:
        from sample import load_checkpoint

        ck = load_checkpoint(args.ckpt)
        cfg = importlib.import_module(f"configs.{ck['config']}").model
        model = GPT(cfg).eval()
        model.load_state_dict(ck["model"])
        encode, name = bebut_tok.encode, f"{args.ckpt} (config {ck['config']}, step {ck.get('step')})"
        del ck
    else:
        model, encode = load_hf_model(args.hf, max_seq_len=max(args.ctx, 2048))
        cfg, name = model.cfg, args.hf
    params = sum(p.numel() for p in model.parameters())
    print(f"모델: {name} — {params / 1e9:.2f}B 파라미터, {time.time() - t0:.0f}초", flush=True)

    docs = _stream("HuggingFaceFW/fineweb-2", "kor_Hang", "test", int(args.kb * 1000))
    chunks = [c for d in docs for c in chunks_of(encode(d), args.ctx)]
    chunks = chunks[: args.max_chunks] if args.max_chunks else chunks
    if not chunks:
        raise SystemExit(f"글이 너무 짧다: {args.ctx}토큰을 채우는 문서가 없다 (--kb를 늘리거나 --ctx를 줄이세요)")
    print(f"글: 문서 {len(docs)}개 → {args.ctx}토큰 묶음 {len(chunks)}개", flush=True)

    t = time.time()
    ref_nll, ref_top1 = torch_reference(model, chunks)
    results = {"torch_fp32": {**summarize(ref_nll, ref_top1), "seconds": round(time.time() - t)}}
    print(f"기준 (PyTorch fp32): ppl {results['torch_fp32']['ppl']:.3f}", flush=True)

    check = {}
    if args.torch_check:
        t = time.time()
        check = torch_dequant_scores(model, chunks, [q for q in args.types if q in ("q8_0", "q4_0")])
        for q, (nll, top1) in check.items():
            r = summarize(nll, top1, ref_nll, ref_top1)
            print(f"교차 확인 (PyTorch, {q} 가중치): ppl {r['ppl']:.3f} ({r['ppl_change_pct']:+.2f}%)", flush=True)
        print(f"  {time.time() - t:.0f}초", flush=True)

    sd = model.state_dict()
    with tempfile.TemporaryDirectory(dir=os.environ.get("BEBUT_TMPDIR")) as d:
        paths = {}
        exports = list(args.types) + (["f16"] if args.llama_types and "f16" not in args.types else [])
        for q in exports:  # 모델이 메모리에 있을 때 다 내보내고, llama.cpp를 올리기 전에 지운다
            paths[q] = os.path.join(d, f"m_{q}.gguf")
            export_gguf(sd, cfg, bebut_tok, paths[q], q, chat=False, name="quant-quality", write_modelfile=False)
        del sd, model
        gc.collect()
        jobs = [(q, q) for q in args.types] + [(f"llama:{q}", q) for q in args.llama_types]
        for label, q in jobs:
            t = time.time()
            if label.startswith("llama:"):
                paths[label] = llama_quantize(paths["f16"], os.path.join(d, f"llama_{q}.gguf"), q, args.threads)
            nll, top1 = llama_scores(paths[label], chunks, args.ctx, args.threads)
            results[label] = r = {**summarize(nll, top1, ref_nll, ref_top1), "bytes": os.path.getsize(paths[label]),
                                  "bits_per_weight": 8 * os.path.getsize(paths[label]) / params,
                                  "seconds": round(time.time() - t)}
            print(f"{label:>12}: {r['bytes'] / 1e9:.2f}GB | ppl {r['ppl']:.3f} ({r['ppl_change_pct']:+.2f}%) | "
                  f"토큰별 손실 차이 {r['mean_abs_nll_diff']:.4f} | 1순위 일치 {r['top1_agreement'] * 100:.1f}%",
                  flush=True)
            if label in check:  # 같은 양자화 가중치의 PyTorch 결과, 그리고 llama.cpp가 그것과 얼마나 같은가
                r["torch_dequant"] = summarize(*check[label], ref_nll, ref_top1)
                r["vs_torch_dequant"] = {"mean_abs_nll_diff": float(np.abs(nll - check[label][0]).mean()),
                                         "top1_agreement": float((top1 == check[label][1]).mean())}
                print(f"{'':>12}  PyTorch에 같은 가중치: ppl {r['torch_dequant']['ppl_change_pct']:+.2f}% | "
                      f"llama.cpp와 토큰별 손실 차이 {r['vs_torch_dequant']['mean_abs_nll_diff']:.4f} · "
                      f"1순위 일치 {r['vs_torch_dequant']['top1_agreement'] * 100:.1f}%", flush=True)
            if label != "f16" or not args.llama_types:  # F16은 llama.cpp 양자화의 원본으로 남겨 둔다
                os.remove(paths[label])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": name, "params": params, "ctx": args.ctx, "chunks": len(chunks),
                       "text": "FineWeb-2 ko test", "results": results}, f, ensure_ascii=False, indent=1)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
