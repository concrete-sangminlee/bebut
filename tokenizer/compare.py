"""토크나이저 효율 비교 — 같은 한국어 텍스트를 몇 토큰으로 쪼개는가.

토큰이 적을수록 같은 문맥 길이(2048)에 더 많은 글이 들어가고, 학습·추론 모두 글자당
계산이 줄어든다. 직접 학습한 64K 토크나이저를 공개 모델들의 토크나이저와 같은 텍스트로 잰다.

공정성: 우리 토크나이저는 FineWeb-2 ko train split 앞쪽 400MB, 위키백과 앞쪽 50MB,
FineWeb-Edu 앞쪽 50MB로 학습했다. 그래서 비교 텍스트는
  웹    FineWeb-2 ko **test** split (학습에 안 쓴 분할)
  위키  위키백과에서 앞쪽 60MB를 건너뛴 뒤
  시험  KMMLU test 문항 + 보기 (학습 데이터에 없는 영역)
  영어  FineWeb-Edu에서 앞쪽 60MB를 건너뛴 뒤
를 쓴다. 지표는 글자/토큰(클수록 좋음). special token은 붙이지 않는다.

공개 토크나이저는 HF `tokenizers` 라이브러리로 읽는다 (transformers와 함께 설치된다 —
저장소 의존성은 아니고 이 비교에만 쓴다).

사용: python -m tokenizer.compare --mb 2 --out logs/tokenizer_compare.json
"""

import argparse
import json
import os
import time

from .bpe import ByteBPETokenizer

PUBLIC = [  # (표시 이름, HF 저장소)
    ("GPT-4o (o200k)", "Xenova/gpt-4o"),
    ("GPT-4 (cl100k)", "Xenova/gpt-4"),
    ("Qwen2.5", "Qwen/Qwen2.5-0.5B"),
    ("Llama 3", "beomi/Llama-3-Open-Ko-8B"),
    ("Mistral-Nemo (Tekken)", "mistralai/Mistral-Nemo-Instruct-2407"),
    ("EXAONE 3.5", "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"),
    ("Kanana", "kakaocorp/kanana-nano-2.1b-base"),
    ("SOLAR", "upstage/SOLAR-10.7B-v1.0"),
    ("Polyglot-ko", "EleutherAI/polyglot-ko-1.3b"),
    ("KoGPT2", "skt/kogpt2-base-v2"),
]


def _stream(name, config, split, budget, skip_bytes=0):
    from datasets import load_dataset

    skipped = got = 0
    out = []
    for row in load_dataset(name, config, split=split, streaming=True):
        text = row["text"]
        n = len(text.encode("utf-8"))
        if skipped < skip_bytes:
            skipped += n
            continue
        out.append(text)
        got += n
        if got >= budget:
            break
    return out


def _kmmlu(budget):
    from datasets import get_dataset_config_names, load_dataset

    out, got = [], 0
    subjects = get_dataset_config_names("HAERAE-HUB/KMMLU")
    per = max(1, int(budget / len(subjects)))
    for sub in subjects:
        sub_got = 0
        for r in load_dataset("HAERAE-HUB/KMMLU", sub, split="test"):
            text = r["question"] + "\n" + "\n".join(r[k] for k in "ABCD")
            out.append(text)
            sub_got += len(text.encode("utf-8"))
            if sub_got >= per:
                break
        got += sub_got
    return out


def load_domains(mb: float):
    b = int(mb * 1e6)
    return {
        "웹": _stream("HuggingFaceFW/fineweb-2", "kor_Hang", "test", b),
        "위키": _stream("wikimedia/wikipedia", "20231101.ko", "train", b, skip_bytes=60e6),
        "시험(KMMLU)": _kmmlu(b),
        "영어": _stream("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", b, skip_bytes=60e6),
    }


def count_tokens(encode, texts) -> int:
    return sum(len(encode(t)) for t in texts)


def compare(domains, tokenizers):
    """{토크나이저: {"vocab": V, 영역: 글자/토큰}}."""
    chars = {d: sum(len(t) for t in texts) for d, texts in domains.items()}
    res = {}
    for name, (vocab, encode) in tokenizers.items():
        r = {"vocab": vocab}
        for d, texts in domains.items():
            r[d] = round(chars[d] / max(1, count_tokens(encode, texts)), 3)
        res[name] = r
    return res, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=2.0, help="영역별 텍스트 양 (MB)")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    t0 = time.time()
    domains = load_domains(args.mb)
    print(f"텍스트 준비 ({time.time() - t0:.0f}s): " +
          ", ".join(f"{d} {sum(len(t) for t in v) / 1e6:.2f}M자" for d, v in domains.items()), flush=True)

    ours = ByteBPETokenizer.load(args.tokenizer)
    toks = {"bebut (이 저장소)": (ours.vocab_size, ours.encode)}
    for label, repo in PUBLIC:
        try:
            t = Tokenizer.from_pretrained(repo)
        except Exception as e:  # 비공개 전환·네트워크 문제 — 그 하나만 빼고 계속한다
            print(f"  건너뜀 {label}: {type(e).__name__}")
            continue
        toks[label] = (t.get_vocab_size(), lambda s, t=t: t.encode(s, add_special_tokens=False).ids)

    res, chars = compare(domains, toks)
    names = list(domains)
    print(f"\n글자/토큰 (클수록 같은 글을 적은 토큰으로) — 영역별 {args.mb}MB\n")
    print(f"{'토크나이저':<24}{'vocab':>9}" + "".join(f"{d:>12}" for d in names))
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["웹"]):
        print(f"{name:<24}{r['vocab']:>9,}" + "".join(f"{r[d]:>12.2f}" for d in names))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"mb_per_domain": args.mb, "chars": chars, "chars_per_token": res,
                       "seconds": round(time.time() - t0)}, f, ensure_ascii=False, indent=1)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
