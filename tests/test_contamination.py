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


def test_copa_wrong_answer_sharing_words_is_not_flagged():
    """두 답이 앞부분을 공유하면(머리를 잘랐다 / 길렀다) 전제 + 오답 글이 키의 절반을 채워 오염으로
    잡혔다. 정답에만 있는 단어가 겹쳐야 한다."""
    from evals.contamination import copa_must

    row = {"premise": "여자는 긴 머리가 거추장스럽다고 느꼈다.", "alternative_1": "여자는 머리를 잘랐다.",
           "alternative_2": "여자는 머리를 길렀다.", "label": 0, "question": "결과"}

    def items():
        yield "copa", "kobest/copa/7", copa_forms(row), None, copa_must(row)

    wrong = [("web", 0, f"{row['premise']} 그래서 {row['alternative_2']}")]
    right = [("web", 0, f"{row['premise']} 그래서 {row['alternative_1']}")]
    assert run(wrong, items=items)[1] == [] and run(right, items=items)[1] == ["kobest/copa/7"]


def test_passage_exposure_is_reported_but_not_excluded():
    passage = ("피타고라스 정리는 직각 삼각형의 빗변의 제곱이 두 직각변의 제곱의 합과 같다는 "
               "정리이다. 유클리드 기하학에서 가장 잘 알려진 정리 가운데 하나이다.")

    def items():
        yield "boolq", "kobest/boolq/0", "피타고라스 정리는 모든 삼각형에서 성립하는가"
        yield "boolq_passage", "kobest/boolq/0", passage

    res, ids = run([("wiki", 5, passage)], items=items)
    assert res["boolq_passage"]["contaminated"] == 1 and res["boolq"]["any_match"] == 0
    assert ids == []  # 지문 노출은 문항 노출이 아니다 — 채점에서 빼지 않는다


def mc(item_id, q, answer, options, bench="kmmlu"):
    return bench, item_id, q, (answer, options)


BIO = ("담자균류", ["자낭균류", "담자균류", "조균류", "불완전균류"])


def test_window_boundary():
    """정답은 겹친 줄기 바로 뒤 WINDOW(200) 단어 안에 있어야 한다."""
    def items():
        yield mc("kmmlu/Biology/0", KMMLU_Q, *BIO)

    assert run([("web", 0, f"{KMMLU_Q} " + "잡문 " * 199 + "담자균류")], items=items)[1] == ["kmmlu/Biology/0"]
    assert run([("web", 0, f"{KMMLU_Q} " + "잡문 " * 200 + "담자균류")], items=items)[1] == []


def test_occurrences_are_checked_where_they_are():
    """확인은 문서 안 출현마다 한다 (리뷰에서 나온 놓침·오탐 사례)."""
    # 줄기가 WINDOW보다 길어도 줄기 끝 바로 뒤를 본다
    long_q = " ".join(f"단어{k}번" for k in range(260))

    def long_items():
        yield mc("haerae/RC/0", long_q, "정답보기", ["정답보기", "오답하나", "오답둘", "오답셋"], "haerae")

    assert run([("web", 0, f"{long_q} 1 정답보기 2 오답하나")], items=long_items)[1] == ["haerae/RC/0"]

    # 같은 줄기가 먼저 다른 보기로 나오고, 진짜 문항은 한참 뒤에 나와도 잡는다
    def bio():
        yield mc("kmmlu/Biology/0", KMMLU_Q, *BIO)

    filler = " ".join(f"잡문{k}" for k in range(250))
    doc = f"{KMMLU_Q} ① 접합균류 ② 난균류 {filler} {KMMLU_Q} ① 자낭균류 ② 담자균류 ③ 조균류"
    assert run([("web", 0, doc)], items=bio)[1] == ["kmmlu/Biology/0"]

    # 여러 문항이 공유하는 상투구가 앞에서 수없이 반복돼도 뒤의 진짜 문항을 잡는다
    bp = "임대차에 관한 설명으로 옳은 것은 다툼이 있으면 판례에 따름"
    q2 = f"갑은 을 소유의 건물을 보증금 1억원에 임차하여 인도받고 전입신고를 마쳤다 {bp}"

    def law():
        yield mc("kmmlu/Law/0", q2, "확정일자를 받은 경우 우선변제권이 있다",
                 ["확정일자를 받은 경우 우선변제권이 있다", "대항력이 없다", "차임 증액은 언제나 가능하다"])
        yield mc("kmmlu/Law/1", f"다른 내용의 임대차 사례 문제 줄기 여기에 {bp}", "가나다라",
                 ["가나다라", "마바사", "아자차"])

    others = " ".join(f"{k}번 문제 사례 {k} {bp} ① 보기{k}가 ② 보기{k}나" for k in range(300))
    res, ids = run([("web", 0, f"{others} {q2} ① 확정일자를 받은 경우 우선변제권이 있다")], items=law)
    assert ids == ["kmmlu/Law/0"]  # 상투구만 공유한 Law/1은 질문 겹침도 오염도 아니다

    # 겹침을 만든 문서와 정답이 나온 자리가 달라선 안 된다: A는 줄기 전체(다른 보기),
    # B는 키 하나 + 정답 → 오염 아님 (예전엔 B의 정답으로 오염 판정)
    q3 = "다음 중 조선 후기 실학자 정약용이 저술한 책으로 지방 관리의 도리를 다룬 것은 무엇인가"

    def hist():
        yield mc("kmmlu/Korean-History/0", q3, "목민심서", ["목민심서", "열하일기", "북학의", "반계수록"])

    doc_a = ("webA", 0, f"{q3} ① 경세유표 ② 흠흠신서 ③ 여유당전서 ④ 아방강역고")
    doc_b = ("wiki", 7, "정약용은 " + " ".join(words(q3)[2:10]) + " 이야기와 관련하여 목민심서를 남겼다")
    res, ids = run([doc_a, doc_b], items=hist)
    assert ids == [] and res["kmmlu"]["flagged"] == 1


def test_option_boundaries():
    q4 = "어떤 회사의 당기 매출액과 매출원가가 다음과 같을 때 매출총이익은 얼마인가 단위는 만원이다"

    def acc():
        yield mc("kmmlu/Accounting/0", q4, "120", ["100", "120", "140", "160"])

    assert run([("web", 0, f"{q4} ① 1200 ② 1400 ③ 1600 ④ 1800")], items=acc)[1] == []  # 120 ≠ 1200
    assert run([("web", 0, f"{q4} ① 100 ② 120 ③ 140 ④ 160")], items=acc)[1] == ["kmmlu/Accounting/0"]
    # ㄱ·ㄴ 조합 보기는 확인에 못 쓴다 → 질문이 그대로 있어도 '확인 불가'로 따로 센다
    assert choice_check("옳은 것을 모두 고른 것은", "ㄱ, ㄷ", ["ㄱ, ㄴ", "ㄱ, ㄷ", "ㄴ, ㄷ"]) == (None, ())

    def jamo():
        yield mc("kmmlu/Law/3", KMMLU_Q, "ㄱ, ㄷ", ["ㄱ, ㄴ", "ㄱ, ㄷ", "ㄴ, ㄷ", "ㄱ, ㄴ, ㄷ"])

    res, ids = run([("web", 0, f"{KMMLU_Q} ① ㄱ, ㄴ ② ㄱ, ㄷ ③ ㄴ, ㄷ")], items=jamo)
    assert ids == [] and res["kmmlu"]["unverifiable"] == 1


def test_save_and_load_found_merge(tmp_path):
    from evals.contamination import index_digest, load_found, save_found

    per_item, index = build_index(evals())
    digest = index_digest(index, 0.5)
    parts = [PLANTED[:2], PLANTED[2:]]  # 나눠 훑은 두 결과를 합치면 한 번에 훑은 것과 같다
    for k, rows in enumerate(parts):
        save_found(tmp_path / f"{k}.json", digest, *scan(iter(rows), index, workers=1))
    found, verified, counts = {}, {}, {}
    load_found([tmp_path / "0.json", tmp_path / "1.json"], digest, found, verified, counts)
    assert counts == {"sft": 2, "textbook": 2}
    assert report(per_item, index, found, verified)[1] == run(PLANTED)[1]
    with pytest.raises(SystemExit):  # 다른 설정으로 만든 결과는 섞지 않는다
        load_found([tmp_path / "0.json"], index_digest(index, 0.6), {}, {}, {})
