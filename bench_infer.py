"""추론 속도 측정 — docs/INFERENCE.md의 표를 이 스크립트 하나로 다시 만든다.

설정의 실제 크기 모델을 무작위 가중치로 만들어 잰다(디코드 속도는 가중치 값과 무관하다).
예전에는 이 수치들을 재현할 스크립트가 없었고, 한 번은 다른 작업이 CPU를 쓰는 중에 재서
30~40% 낮게 적은 적이 있다. 그래서 값마다 여러 번 재 중앙값을 쓰고, 측정 환경(CPU · 스레드 ·
시작 시점의 부하)을 결과에 함께 남긴다.

재는 것:
  디코드       PyTorch fp32 · bf16 (greedy, 첫 토큰 이후 토큰당 속도)
  첫 토큰까지   긴 기록 + 새 질문에서, 매 턴 전부 계산 vs 프리픽스 KV 캐시 재사용
  GGUF         llama.cpp로 F16 · Q8_0 · Q4_0의 크기와 디코드 속도 (llama-cpp-python이 있을 때)
  샘플러       64K vocab에서 토큰 하나를 뽑는 시간 (greedy / top-p, 뾰족한 분포와 평평한 분포)
  문맥 창      창이 찬 뒤 턴마다 새로 계산하는 토큰 수 (대화 80턴 모의, 토크나이저만 쓴다)

사용:
  python bench_infer.py --config base_1b --out logs/inference_bench.json   # 1.2B, 메모리 약 10GB, 15분
  python bench_infer.py --config mid_125m                                  # 작은 모델로 빠르게
"""

import argparse
import gc
import importlib
import json
import os
import platform
import random
import statistics
import tempfile
import time

import torch

from chat import encode_prompt, fit_messages
from model import GPT, KVCache, sample_next
from sample import cpu_has_fast_bf16
from tokenizer.bpe import ByteBPETokenizer


def env_info(threads):
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    info = {"cpu": cpu, "cores": os.cpu_count(), "threads": threads, "torch": torch.__version__,
            "python": platform.python_version(), "cpu_bf16": cpu_has_fast_bf16()}
    if hasattr(os, "getloadavg"):
        info["loadavg_at_start"] = round(os.getloadavg()[0], 2)  # 높으면 다른 작업이 CPU를 쓰는 중이다
    return info


def median(fn, reps):
    return statistics.median(fn() for _ in range(reps))


def decode_tps(model, ctx, n, reps):
    """디코드만 잰다 (프롬프트 prefill 제외): 첫 토큰이 나온 뒤 n토큰의 속도."""
    idx = torch.randint(0, model.cfg.vocab_size, (1, ctx), generator=torch.Generator().manual_seed(0))
    model.generate(idx, 2, temperature=0.0)  # 워밍업

    def once():
        it = model.generate_stream(idx, n + 1, temperature=0.0)
        next(it)
        t = time.perf_counter()
        for _ in it:
            pass
        return n / (time.perf_counter() - t)

    return median(once, reps)


def ttft(model, history, question, reps):
    """기록 history토큰 + 새 질문 question토큰에서 첫 토큰까지의 시간(초): (캐시 없이, 프리픽스 캐시)."""
    g = torch.Generator().manual_seed(1)
    hist = torch.randint(0, model.cfg.vocab_size, (1, history), generator=g)
    nxt = torch.cat([hist, torch.randint(0, model.cfg.vocab_size, (1, question), generator=g)], dim=1)

    def cold():
        t = time.perf_counter()
        next(model.generate_stream(nxt, 1, temperature=0.0))
        return time.perf_counter() - t

    def warm():
        cache = model.new_cache()
        list(model.generate_stream(hist, 1, temperature=0.0, cache=cache))  # 지난 턴
        t = time.perf_counter()
        next(model.generate_stream(nxt, 1, temperature=0.0, cache=cache))
        return time.perf_counter() - t

    return median(cold, reps), median(warm, reps)


def sampler_ms(vocab, reps=200):
    """토큰 하나를 고르는 시간(ms). 학습된 언어모델의 분포는 뾰족하다(Zipf) — 무작위 가중치 모델의
    평평한 분포로 재면 top-p가 늘 전체 정렬로 넘어가 실제보다 느리게 나온다."""
    g = torch.Generator().manual_seed(0)
    peaked = (-1.1 * torch.log(torch.arange(1, vocab + 1).float()))[torch.randperm(vocab, generator=g)][None]
    flat = torch.randn(1, vocab, generator=g) * 0.5

    def ms(logits, **kw):
        sample_next(logits, generator=g, **kw)
        t = time.perf_counter()
        for _ in range(reps):
            sample_next(logits, generator=g, **kw)
        return round((time.perf_counter() - t) / reps * 1e3, 3)

    return {"greedy": ms(peaked, temperature=0.0),
            "top_p_0.95_peaked": ms(peaked, temperature=0.7, top_p=0.95),
            "top_p_0.95_flat": ms(flat, temperature=0.7, top_p=0.95),
            "top_p_1.0": ms(peaked, temperature=0.7, top_p=1.0)}


def window_reuse(tok, max_seq_len=2048, max_tokens=256, turns=80, seed=0):
    """대화가 문맥 창을 채운 뒤, 턴마다 새로 계산해야 하는 토큰 수 (모델 없이 캐시 장부만)."""
    rng = random.Random(seed)
    words = ("대한민국 헌법 민주공화국 주권은 국민에게 있고 모든 권력은 국민으로부터 나온다 "
             "서울 인공지능 데이터 학습").split()

    def sent(n):
        return " ".join(rng.choice(words) for _ in range(n)) + "."

    budget = max(max_seq_len // 2, max_seq_len - max_tokens)
    cache, msgs = KVCache(1, max_seq_len), [{"role": "system", "content": "너는 친절한 비서다."}]
    reused = recomputed = n = 0
    for _ in range(turns):
        msgs.append({"role": "user", "content": sent(rng.randint(8, 25))})
        full = len(encode_prompt(tok, msgs)) > budget  # 기록 전체가 창을 넘었다 = 오래된 턴을 버리는 구간
        ids = fit_messages(tok, msgs, budget)
        k = cache.reuse(ids)
        reply = sent(rng.randint(30, 70))
        cache.tokens = ids + tok.encode(reply)
        cache.pos = len(cache.tokens)
        msgs.append({"role": "assistant", "content": reply})
        if full:
            reused, recomputed, n = reused + k, recomputed + len(ids) - k, n + 1
    return {"turns_measured": n, "recomputed_per_turn": round(recomputed / max(1, n)),
            "reused_per_turn": round(reused / max(1, n))}


def llama_decode_tps(llm, prompt, n):
    """llama.cpp 디코드 속도: 첫 토큰이 나온 뒤 n토큰 (프롬프트 처리 제외)."""
    k, t = 0, time.perf_counter()
    for _ in llm.generate(prompt, top_k=1, reset=True):
        k += 1
        if k == 1:
            t = time.perf_counter()
        if k > n:
            break
    return n / (time.perf_counter() - t)


def gguf_bench(state_dict, cfg, tok, types, ctx, n, reps, threads):
    """llama.cpp로 GGUF 타입별 크기와 디코드 속도. llama-cpp-python이 없으면 None."""
    try:
        from llama_cpp import Llama
    except ImportError:
        return None
    from export_gguf import export_gguf

    out = {}
    with tempfile.TemporaryDirectory(dir=os.environ.get("BEBUT_TMPDIR")) as d:
        paths = {q: export_gguf(state_dict, cfg, tok, os.path.join(d, f"m_{q}.gguf"), q, name="bench",
                                write_modelfile=False) for q in types}
        del state_dict
        gc.collect()
        prompt = list(range(300, 300 + ctx))
        for q, path in paths.items():
            llm = Llama(model_path=path, n_ctx=ctx + n + 8, n_threads=threads, verbose=False)
            llama_decode_tps(llm, prompt, 2)  # 워밍업
            tps = statistics.median(llama_decode_tps(llm, prompt, n) for _ in range(reps))
            out[q] = {"bytes": os.path.getsize(path), "decode_tok_s": round(tps, 2)}
            print(f"  GGUF {q}: {out[q]['bytes'] / 1e9:.2f}GB, {out[q]['decode_tok_s']} tok/s", flush=True)
            llm.close()
            gc.collect()
            os.remove(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base_1b")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--reps", type=int, default=3, help="값마다 재는 횟수 (중앙값을 쓴다)")
    ap.add_argument("--ctx", type=int, default=128, help="디코드 측정의 프롬프트 길이")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--history", type=int, default=1000, help="첫 토큰 측정의 기록 길이")
    ap.add_argument("--question", type=int, default=60)
    ap.add_argument("--types", nargs="*", default=["f16", "q8_0", "q4_0"], help="GGUF 타입 (비우면 GGUF 생략)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = importlib.import_module(f"configs.{args.config}").model
    # 작은 설정(문맥이 짧은 테스트용 모델)에서도 돌게 길이를 문맥에 맞춘다
    args.question = min(args.question, cfg.max_seq_len // 4)
    history = min(args.history, cfg.max_seq_len - args.question - 8)
    args.ctx = min(args.ctx, cfg.max_seq_len // 2)
    args.new_tokens = min(args.new_tokens, cfg.max_seq_len - args.ctx - 2)
    res = {"config": args.config, "env": env_info(args.threads), "reps": args.reps,
           "note": "무작위 가중치, 값마다 중앙값. 디코드는 첫 토큰 이후 토큰당 속도"}
    if res["env"].get("loadavg_at_start", 0) > 1.0:
        print(f"경고: 시작 시점 부하 {res['env']['loadavg_at_start']} — "
              "다른 작업이 CPU를 쓰는 중이면 수치가 낮게 나온다")

    model = GPT(cfg).eval()
    res["params"] = sum(p.numel() for p in model.parameters())
    print(f"{args.config}: {res['params'] / 1e6:.0f}M 파라미터, 스레드 {args.threads}", flush=True)

    res["torch"] = {"fp32_decode_tok_s": round(decode_tps(model, args.ctx, args.new_tokens, args.reps), 2)}
    print(f"  PyTorch fp32: {res['torch']['fp32_decode_tok_s']} tok/s", flush=True)

    tok = ByteBPETokenizer.load(args.tokenizer)
    gguf_state = None
    if args.types and tok.vocab_size <= cfg.vocab_size:
        gguf_state = {k: v.clone() for k, v in model.state_dict().items()}  # bf16으로 바꾸기 전의 fp32

    model = model.to(torch.bfloat16)
    res["torch"]["bf16_decode_tok_s"] = round(decode_tps(model, args.ctx, args.new_tokens, args.reps), 2)
    print(f"  PyTorch bf16: {res['torch']['bf16_decode_tok_s']} tok/s", flush=True)
    if not res["env"]["cpu_bf16"]:  # bf16 명령이 없는 CPU에선 실제 기본값인 fp32로 첫 토큰을 잰다
        model = model.to(torch.float32)
    cold, warm = ttft(model, history, args.question, args.reps)
    res["ttft"] = {"dtype": "bf16" if res["env"]["cpu_bf16"] else "fp32", "history_tokens": history,
                   "question_tokens": args.question, "no_cache_s": round(cold, 3), "prefix_cache_s": round(warm, 3)}
    print(f"  첫 토큰까지 ({history}+{args.question}토큰): 캐시 없이 {cold:.2f}s → 프리픽스 캐시 {warm:.2f}s",
          flush=True)
    del model
    gc.collect()

    res["sampler_ms"] = sampler_ms(cfg.vocab_size)
    print(f"  샘플러(ms/토큰): {res['sampler_ms']}", flush=True)
    res["window_reuse"] = window_reuse(tok) if tok.vocab_size > 60000 else None
    if res["window_reuse"]:
        print(f"  문맥 창이 찬 뒤 턴당 새로 계산 {res['window_reuse']['recomputed_per_turn']}토큰", flush=True)

    res["gguf"] = (gguf_bench(gguf_state, cfg, tok, args.types, args.ctx, args.new_tokens, args.reps, args.threads)
                   if gguf_state is not None else None)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
