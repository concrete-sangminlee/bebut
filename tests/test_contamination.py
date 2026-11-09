"""오염 검사 도구 검증: 심어 둔 문항은 잡고, 무관한 문항은 잡지 않는다."""

from evals.contamination import build_index, item_keys, report, scan, words

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
    assert len(item_keys(words(KMMLU_Q))) == len(words(KMMLU_Q)) - 7
    assert item_keys(words(SHORT_Q)) == ["비가 많이 와서 땅이 젖었다"]
    assert item_keys(["a", "b"]) == []


def run(train_rows, workers):
    per_item, keys = build_index(evals())
    found, counts = scan(iter(train_rows), keys, workers=workers)
    return report(per_item, found, threshold=0.5)


def test_planted_item_is_flagged_with_evidence():
    half = " ".join(words(OTHER_Q)[:8])  # 앞 8단어만 → 겹침 비율이 낮다
    train = [("sft", 0, "아무 관련 없는 문장입니다. " * 5),
             ("sft", 1, f"질문: {KMMLU_Q}? 정답은 담자균류입니다."),  # 문장부호가 달라도 잡힌다
             ("textbook", 2, f"오늘 {half} 라는 문장이 나왔다"),
             ("textbook", 3, "그날 비가 많이 와서 땅이 젖었다 그래서 집에 있었다")]
    for workers in (1, 2):
        res, flagged = run(train, workers)
        assert flagged == ["kmmlu/Biology/0", "kobest/copa/7"]
        assert res["kmmlu"]["examples"][0]["evidence"] == "sft#1"
        assert res["kmmlu"]["examples"][0]["overlap"] == 1.0
        assert res["haerae"]["flagged"] == 0 and res["haerae"]["any_match"] == 1  # 일부만 겹침
        assert res["copa"]["items"] == 2 and res["copa"]["checkable"] == 1


def test_clean_data_flags_nothing():
    res, flagged = run([("sft", i, f"전혀 다른 문장 {i}번입니다 " * 3) for i in range(50)], 1)
    assert flagged == [] and all(r["any_match"] == 0 for r in res.values())
