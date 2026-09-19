"""평가 문제가 학습 데이터(KULLM v2, KoAlpaca)에 들어 있는지 검사한다.

이 두 데이터셋은 SFT뿐 아니라 annealing 구간 사전학습에도 들어갔다. 여기 겹치는
문제로 평가하면 '실력'이 아니라 '기억'을 재게 된다 (SFT 검증 loss에서 실제로 겪었다).

문자 3-gram Jaccard 유사도로 가장 비슷한 학습 예제를 찾아, 0.5 이상이면 오염으로 본다.
"""
import json
import re

from datasets import load_dataset

def grams(s):
    s = re.sub(r"[\s\W_]+", "", s)
    return {s[i:i + 3] for i in range(len(s) - 2)}

qs = json.load(open("evals/heldout_ko.json"))
train = []
for name, fi, fin in [("beomi/KoAlpaca-v1.1a", "instruction", None),
                      ("nlpai-lab/kullm-v2", "instruction", "input")]:
    for r in load_dataset(name, split="train"):
        t = r[fi] or ""
        if fin and r.get(fin):
            t += " " + r[fin]
        if t:
            train.append(t)
print(f"학습 instruction {len(train):,}개와 비교\n")

tg = [grams(t) for t in train]
index = {}
for i, g in enumerate(tg):
    for x in g:
        index.setdefault(x, []).append(i)

flagged = 0
for q in qs:
    g = grams(q["q"])
    cand = {}
    for x in g:
        for i in index.get(x, ()):
            cand[i] = cand.get(i, 0) + 1
    best, bi = 0.0, None
    for i, inter in cand.items():
        j = inter / len(g | tg[i])
        if j > best:
            best, bi = j, i
    mark = "  ← 오염 의심" if best >= 0.5 else ""
    flagged += best >= 0.5
    print(f"{best:.2f} | {q['q'][:28]:<30} | {train[bi][:40] if bi is not None else ''}{mark}".replace("\n", " "))
print(f"\n오염 의심 {flagged}/{len(qs)}개")
