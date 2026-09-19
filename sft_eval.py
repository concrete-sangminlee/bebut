"""SFT 평가 — 베이스 / SFT 에폭0 / SFT 최종을 같은 조건으로 비교.

1) 대화 평가: evals/heldout_ko.json (학습 데이터와 겹치지 않음을 check_contamination.py로 확인).
   greedy 생성 후 정답 키워드 포함 여부로 채점. '오염대조'는 학습 데이터에 거의 같은 문제가
   있는 것이라 채점에서 빼고 따로 보고한다 — 안 본 문제와의 차이가 기억 효과다.
   정상 종료율(<|im_end|>로 스스로 멈췄는가)과 반복률도 잰다.
2) KOBEST 전체 문항 (사전학습 최종 평가와 같은 2-shot 형식) — SFT가 기존 능력을 깎았는지 본다.

사용: python sft_eval.py  →  logs/sft_eval.json, logs/sft_eval_answers.md
"""
import json
import re
import time

import torch

import configs.base_1b as conf
from eval import FORMATTERS, eval_kobest
from model import GPT
from tokenizer.bpe import ByteBPETokenizer

DEV = "cuda"
MODELS = [("베이스", "checkpoints/base_1b/latest.pt"),
          ("SFT 에폭0", "checkpoints/sft_1b/epoch0.pt"),
          ("SFT 최종", "checkpoints/sft_1b/latest.pt")]
MAX_NEW = 150

tok = ByteBPETokenizer.load("tokenizer/tokenizer.json")
END = tok.special_tokens["<|im_end|>"]
QS = json.load(open("evals/heldout_ko.json"))


def correct(ans, keys):
    for k in keys:
        if k.isdigit():  # 숫자는 '17' 안의 '7'처럼 걸리지 않게 경계를 본다
            if re.search(rf"(?<!\d){k}(?!\d)", ans):
                return True
        elif k in ans:
            return True
    return False


def repetitive(ans):
    """8자 이상 구절이 3번 이상 반복되면 반복 루프로 본다."""
    s = re.sub(r"\s+", " ", ans)
    for L in (8, 12, 20):
        seen = {}
        for i in range(len(s) - L + 1):
            seen[s[i:i + L]] = seen.get(s[i:i + L], 0) + 1
            if seen[s[i:i + L]] >= 3:
                return True
    return False


results, answers_md = {}, ["# SFT 평가 — 모델별 답변\n"]
for label, path in MODELS:
    t0 = time.time()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(conf.model)
    model.load_state_dict(ck["model"])
    del ck
    model = model.to(DEV).eval()
    print(f"\n===== {label} =====", flush=True)
    answers_md.append(f"\n## {label}\n")

    by_type, stopped, rep, rows = {}, 0, 0, []
    for q in QS:
        prompt = f"<|im_start|>user\n{q['q']}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok.encode(prompt, allowed_special=True)
        out = model.generate(torch.tensor([ids], device=DEV), MAX_NEW, temperature=0.0, eos_id=END)
        gen = out[0, len(ids):].tolist()
        ended = END in gen
        ans = tok.decode([t for t in gen if t != END]).strip()
        stopped += ended
        r = repetitive(ans)
        rep += r
        ok = correct(ans, q["a"]) if q["a"] else None
        if ok is not None:
            by_type.setdefault(q["type"], []).append(ok)
        rows.append({"type": q["type"], "q": q["q"], "answer": ans, "correct": ok,
                     "stopped": ended, "repetitive": r})
        mark = "" if ok is None else ("O " if ok else "X ")
        answers_md.append(f"- **[{q['type']}] {q['q']}** {mark}\n  > {ans[:300].replace(chr(10), ' ')}\n")

    acc = {t: round(100 * sum(v) / len(v), 1) for t, v in by_type.items()}
    held = [ok for t, v in by_type.items() if t != "오염대조" for ok in v]
    r = {"heldout_acc": round(100 * sum(held) / len(held), 1), "by_type": acc,
         "stop_rate": round(100 * stopped / len(QS), 1), "repeat_rate": round(100 * rep / len(QS), 1),
         "kobest": {}, "answers": rows}
    print(f"  대화 정답률(비오염) {r['heldout_acc']}% | 유형별 {acc}", flush=True)
    print(f"  스스로 멈춤 {r['stop_rate']}% | 반복 루프 {r['repeat_rate']}%", flush=True)

    for task in FORMATTERS:
        a, n, preds = eval_kobest(model, tok, task, 2, DEV, None)
        r["kobest"][task] = {"acc": round(a * 100, 1), "n": n,
                             "preds": {str(k): v for k, v in sorted(preds.items())}}
        print(f"  {task:<10} {a*100:5.1f}% (n={n}) 예측분포 {dict(sorted(preds.items()))}", flush=True)

    r["seconds"] = round(time.time() - t0)
    results[label] = r
    del model
    torch.cuda.empty_cache()

json.dump(results, open("logs/sft_eval.json", "w"), ensure_ascii=False, indent=2)
open("logs/sft_eval_answers.md", "w").write("\n".join(answers_md))
print("\n저장: logs/sft_eval.json, logs/sft_eval_answers.md", flush=True)
