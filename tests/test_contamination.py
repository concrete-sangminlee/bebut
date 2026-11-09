"""오염 검사 도구 검증: 심어 둔 문항은 잡고, 무관한 문항은 잡지 않는다."""

import pytest

from eval import format_copa
from evals.contamination import (build_index, copa_forms, item_keys, report, scan, words)

KMMLU_Q = "사과나무 붉은별무늬병균은 진균 중 어느 균류에 속하는가 다음 중 고르시오"
OTHER_Q = "청동기 시대에 커다란 돌을 쌓아 만든 족장의 무덤을 무엇이라고 하는가"
SHORT_Q = "비가 많이 와서 땅이 젖었다"  # 5단어 — 통째로 있을 때만 걸린다


def evals():
    yield "kmmlu", "kmmlu/Biology/0", KMMLU_Q
    yield "haerae", "haerae/History/3", OTHER_Q
    yield "copa", "kobest/copa/7", SHORT_Q
    yield "copa", "kobest/copa/8", "짧다"  # 검사 불가


def test_keys_and_normalization():
    assert words("Hello, 세계! 12+7=19") == ["hello", "세계", "12", "7", "19"]
    # NFKC: 전각 → 반각. 한자·그리스 문자도 단어로 남는다 (예전엔 버려져 문장이 이어 붙었다)
    assert words("ＡＢＣ１２ 치태(齒苔)와 π") == ["abc12", "치태", "齒苔", "와", "π"]
    assert len(item_keys(words(KMMLU_Q))) == len(words(KMMLU_Q)) - 7
    assert item_keys(words(SHORT_Q)) == ["비가 많이 와서 땅이 젖었다"]
    assert item_keys(["a", "b"]) == []
    # 내용 단어가 모자란 키(숫자 나열·한 글자 변수)는 버린다 — 수식이 우연히 겹치는 것을 막는다
    assert item_keys(words("집합 A={1, 2, 3, 4, 5}, B={1, 2, 3}")) == []
    assert item_keys(words("f(x)f(y) = f(x+y)")) == []


def run(train_rows, workers, start_method=None, items=evals):
    per_item, keys = build_index(items())
    found, counts = scan(iter(train_rows), keys, workers=workers, start_method=start_method)
    return report(per_item, found, threshold=0.5)


PLANTED = [("sft", 0, "아무 관련 없는 문장입니다. " * 5),
           ("sft", 1, f"질문: {KMMLU_Q}? 정답은 담자균류입니다."),  # 문장부호가 달라도 잡힌다
           ("textbook", 2, f"오늘 {' '.join(words(OTHER_Q)[:8])} 라는 문장이 나왔다"),  # 앞 8단어만
           ("textbook", 3, "그날 비가 많이 와서 땅이 젖었다 그래서 집에 있었다")]


@pytest.mark.parametrize("workers,start_method", [(1, None), (2, None), (2, "spawn")])
def test_planted_item_is_flagged_with_evidence(workers, start_method):
    # spawn·forkserver는 전역 변수를 물려받지 않는다 — 색인을 initializer로 넘기지 않으면
    # 워커가 빈 색인으로 돌아 무엇을 넣어도 '오염 0'이 나온다 (예전 구현의 실제 버그)
    res, flagged = run(PLANTED, workers, start_method)
    assert flagged == ["kmmlu/Biology/0", "kobest/copa/7"]
    assert res["kmmlu"]["examples"][0]["evidence"] == "sft#1"
    assert res["kmmlu"]["examples"][0]["overlap"] == 1.0
    assert res["haerae"]["flagged"] == 0 and res["haerae"]["any_match"] == 1  # 일부만 겹침
    partial = res["haerae"]["partial"][0]  # 일부 겹침도 근거를 남긴다
    assert partial["id"] == "haerae/History/3" and partial["evidence"] == "textbook#2"
    assert res["haerae"]["sources"] == {"textbook": 1}
    assert res["copa"]["items"] == 2 and res["copa"]["checkable"] == 1


def test_clean_data_flags_nothing():
    res, flagged = run([("sft", i, f"전혀 다른 문장 {i}번입니다 " * 3) for i in range(50)], 1)
    assert flagged == [] and all(r["any_match"] == 0 for r in res.values())


COPA = [{"premise": "남자의 몸에 그림자가 드리워졌다.", "alternative_1": "해가 떴다.",
         "alternative_2": "비가 왔다.", "label": 0, "question": "원인"},
        {"premise": "아이가 공을 세게 찼다.", "alternative_1": "공이 멀리 날아갔다.",
         "alternative_2": "공이 터졌다.", "label": 0, "question": "결과"}]


def copa_items():
    for i, r in enumerate(COPA):
        yield "copa", f"kobest/copa/{i}", copa_forms(r)


def test_copa_is_caught_in_eval_format_and_natural_order():
    # 전제·답을 그냥 이어 붙여 색인하면 키가 거의 다 이음새를 가로질러, 채점 문맥
    # ("전제 왜냐하면 답")을 그대로 학습 데이터에 넣어도 하나도 못 잡았다
    eval_format = [("sft", i, format_copa(r)[0] + format_copa(r)[1][r["label"]])
                   for i, r in enumerate(COPA)]
    assert run(eval_format, 1, items=copa_items)[1] == ["kobest/copa/0", "kobest/copa/1"]
    natural = [("wiki", 0, "해가 떴다. 그래서 남자의 몸에 그림자가 드리워졌다."),  # 원인 → 결과 순서
               ("wiki", 1, "아이가 공을 세게 찼다. 공이 멀리 날아갔다.")]
    assert run(natural, 1, items=copa_items)[1] == ["kobest/copa/0", "kobest/copa/1"]
    wrong = [("sft", 0, "아이가 공을 세게 찼다 그래서 공이 터졌다")]  # 오답 쪽은 문항 노출이 아니다
    assert run(wrong, 1, items=copa_items)[1] == []


def test_passage_exposure_is_reported_but_not_excluded():
    passage = ("피타고라스 정리는 직각 삼각형의 빗변의 제곱이 두 직각변의 제곱의 합과 같다는 "
               "정리이다. 유클리드 기하학에서 가장 잘 알려진 정리 가운데 하나이다.")

    def items():
        yield "boolq", "kobest/boolq/0", "피타고라스 정리는 모든 삼각형에서 성립하는가"
        yield "boolq_passage", "kobest/boolq/0", passage

    res, flagged = run([("wiki", 5, passage)], 1, items=items)
    assert res["boolq_passage"]["flagged"] == 1 and res["boolq"]["any_match"] == 0
    assert flagged == []  # 지문 노출은 문항 노출이 아니다 — 채점에서 빼지 않는다
