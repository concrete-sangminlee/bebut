"""오염 검사 도구 검증: 심어 둔 문항은 잡고, 무관한 문항은 잡지 않는다."""

import pytest

from eval import format_copa
from evals.contamination import (build_index, choice_check, copa_forms, item_keys, report, scan,
                                 words)

KMMLU_Q = "사과나무 붉은별무늬병균은 진균 중 어느 균류에 속하는가 다음 중 고르시오"
KMMLU_CHOICE = ("담자균류", ["자낭균류", "담자균류", "조균류", "불완전균류"])
OTHER_Q = "청동기 시대에 커다란 돌을 쌓아 만든 족장의 무덤을 무엇이라고 하는가"
SHORT_Q = "비가 많이 와서 땅이 젖었다"  # 5단어 — 통째로 있을 때만 걸린다


def evals():
    yield "kmmlu", "kmmlu/Biology/0", KMMLU_Q, KMMLU_CHOICE
    yield "haerae", "haerae/History/3", OTHER_Q, ("고인돌", ["고인돌", "돌무지무덤", "널무덤", "독무덤"])
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
    # 정답 확인에 쓸 수 없는 보기: 한 글자, 줄기 안에 이미 있는 말
    assert choice_check("TCP의 특징으로 옳은 것은?", "TCP", ["TCP", "UDP 방식", "3"]) == (None, ("udp 방식",))


def run(train_rows, workers=1, start_method=None, items=evals):
    per_item, index = build_index(items())
    found, verified, counts = scan(iter(train_rows), index, workers=workers, start_method=start_method)
    return report(per_item, index, found, verified, threshold=0.5)


PLANTED = [("sft", 0, "아무 관련 없는 문장입니다. " * 5),
           ("sft", 1, f"질문: {KMMLU_Q}? 정답은 담자균류입니다."),  # 문장부호가 달라도 잡힌다
           ("textbook", 2, f"오늘 {' '.join(words(OTHER_Q)[:8])} 라는 문장이 나왔다"),  # 앞 8단어만
           ("textbook", 3, "그날 비가 많이 와서 땅이 젖었다 그래서 집에 있었다")]


@pytest.mark.parametrize("workers,start_method", [(1, None), (2, None), (2, "spawn")])
def test_planted_item_is_flagged_with_evidence(workers, start_method):
    # spawn·forkserver는 전역 변수를 물려받지 않는다 — 색인을 initializer로 넘기지 않으면
    # 워커가 빈 색인으로 돌아 무엇을 넣어도 '오염 0'이 나온다 (예전 구현의 실제 버그)
    res, ids = run(PLANTED, workers, start_method)
    assert ids == ["kmmlu/Biology/0", "kobest/copa/7"]
    ex = res["kmmlu"]["examples"][0]
    assert ex["evidence"] == "sft#1" and ex["overlap"] == 1.0 and ex["answer_follows"]
    assert res["haerae"]["flagged"] == 0 and res["haerae"]["any_match"] == 1  # 일부만 겹침
    partial = res["haerae"]["partial"][0]  # 일부 겹침도 근거를 남긴다
    assert partial["id"] == "haerae/History/3" and partial["evidence"] == "textbook#2"
    assert res["haerae"]["sources"] == {"textbook": 1}
    assert res["copa"]["items"] == 2 and res["copa"]["checkable"] == 1


def test_clean_data_flags_nothing():
    res, ids = run([("sft", i, f"전혀 다른 문장 {i}번입니다 " * 3) for i in range(50)])
    assert ids == [] and all(r["any_match"] == 0 for r in res.values())


def test_multiple_choice_needs_the_answer_after_the_question():
    """시험 줄기는 상투적이라 보기만 바꾼 다른 시험에 같은 줄기가 나온다 (실제 웹 문서에서
    "TCP/IP 프로토콜 구조에 해당하지 않는 것은?"이 보기가 다른 기출로 나왔다). 줄기 뒤에
    정답 보기가 이어져야 오염이다."""
    other_exam = [("web", 0, f"82. {KMMLU_Q}? ① 접합균류 ② 난균류 ③ 변형균류 ④ 점균류")]
    res, ids = run(other_exam)
    assert ids == [] and res["kmmlu"]["flagged"] == 1 and res["kmmlu"]["contaminated"] == 0
    assert res["kmmlu"]["question_only"][0]["answer_follows"] is False
    same_exam = [("web", 0, f"9. {KMMLU_Q}? ㉮ 자낭균류 ㉯ 담자균류 ㉰ 조균류 ㉱ 불완전균류")]
    assert run(same_exam)[1] == ["kmmlu/Biology/0"]
    far = [("web", 0, f"{KMMLU_Q}? " + "다른 내용 " * 150 + "담자균류")]  # 정답이 한참 뒤에만
    assert run(far)[1] == []
    # 정답 표기가 달라도(오타·기호) 다른 보기 3개가 그대로 이어 나오면 같은 문항이다
    typo = [("web", 0, f"{KMMLU_Q}? 1. 자낭균류 2. 담자균 류 3. 조균류 4. 불완전균류")]
    assert run(typo)[1] == ["kmmlu/Biology/0"]


def test_short_answers_fall_back_to_other_options():
    q = "다음 중 대한민국 헌법이 정한 국회의원의 임기로 옳은 것은"

    def items():
        yield "kmmlu", "kmmlu/Law/0", q, ("4", ["2", "3", "4년 임기", "5년 임기"])

    # 정답 "4"는 한 글자라 확인에 못 쓴다 → 다른 보기 중 쓸 수 있는 것(2개)이 다 나와야 한다
    assert run([("web", 0, f"{q} ① 2 ② 3 ③ 4 ④ 6")], items=items)[1] == []
    assert run([("web", 0, f"{q} 1) 2 2) 3 3) 4년 임기 4) 5년 임기")], items=items)[1] == ["kmmlu/Law/0"]


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
    assert run(eval_format, items=copa_items)[1] == ["kobest/copa/0", "kobest/copa/1"]
    natural = [("wiki", 0, "해가 떴다. 그래서 남자의 몸에 그림자가 드리워졌다."),  # 원인 → 결과 순서
               ("wiki", 1, "아이가 공을 세게 찼다. 공이 멀리 날아갔다.")]
    assert run(natural, items=copa_items)[1] == ["kobest/copa/0", "kobest/copa/1"]
    wrong = [("sft", 0, "아이가 공을 세게 찼다 그래서 공이 터졌다")]  # 오답 쪽은 문항 노출이 아니다
    assert run(wrong, items=copa_items)[1] == []


def test_passage_exposure_is_reported_but_not_excluded():
    passage = ("피타고라스 정리는 직각 삼각형의 빗변의 제곱이 두 직각변의 제곱의 합과 같다는 "
               "정리이다. 유클리드 기하학에서 가장 잘 알려진 정리 가운데 하나이다.")

    def items():
        yield "boolq", "kobest/boolq/0", "피타고라스 정리는 모든 삼각형에서 성립하는가"
        yield "boolq_passage", "kobest/boolq/0", passage

    res, ids = run([("wiki", 5, passage)], items=items)
    assert res["boolq_passage"]["contaminated"] == 1 and res["boolq"]["any_match"] == 0
    assert ids == []  # 지문 노출은 문항 노출이 아니다 — 채점에서 빼지 않는다
