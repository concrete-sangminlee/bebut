"""chat 템플릿이 예전 SFT 학습·대화 CLI의 토큰화와 정확히 같은지 검증한다."""

from chat import (StreamDecoder, encode_messages, encode_prompt, fit_messages,
                  instruction_to_messages, render)

QUESTIONS = ["조선을 건국한 사람은 누구야?", "  앞뒤 공백 ", "줄\n바꿈\n\n있는 질문", "emoji 🎉 123,456원"]


def old_sft_example(tok, user, assistant):
    """sft.py의 예전 build_examples 인코딩 그대로."""
    s, e = tok.special_tokens["<|im_start|>"], tok.special_tokens["<|im_end|>"]
    ctx = [s] + tok.encode(f"user\n{user}") + [e] + tok.encode("\n") + [s] + tok.encode("assistant\n")
    ans = tok.encode(assistant) + [e]
    return ctx + ans, [0] * len(ctx) + [1] * len(ans)


def test_single_turn_matches_old_sft_encoding(real_tok):
    for q in QUESTIONS:
        msgs = instruction_to_messages(q, "답변입니다. 이성계입니다.")
        assert encode_messages(real_tok, msgs, add_generation_prompt=False) == \
            old_sft_example(real_tok, msgs[0]["content"], msgs[1]["content"])


def test_prompt_matches_old_chat_cli_and_render(real_tok):
    for q in QUESTIONS:
        msgs = [{"role": "user", "content": q}]
        old = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"  # sample.py/sft_eval.py
        assert render(msgs) == old
        assert encode_prompt(real_tok, msgs) == real_tok.encode(old, allowed_special=True)


def test_instruction_with_input_joined_like_before():
    m = instruction_to_messages("요약해줘", "요약", "긴 글")
    assert m[0]["content"] == "요약해줘\n\n긴 글"


def test_mask_covers_every_assistant_turn(real_tok):
    msgs = [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "반가워요"},
            {"role": "user", "content": "이름은?"}, {"role": "assistant", "content": "bebut"}]
    ids, mask = encode_messages(real_tok, msgs, add_generation_prompt=False)
    picked = real_tok.decode([t for t, m in zip(ids, mask) if m])
    assert picked == "반가워요<|im_end|>bebut<|im_end|>"


def test_fit_messages_drops_oldest_and_keeps_system(real_tok):
    msgs = [{"role": "system", "content": "너는 친절한 비서다."}]
    for i in range(50):
        msgs += [{"role": "user", "content": f"{i}번째 질문입니다 " * 5},
                 {"role": "assistant", "content": f"{i}번째 답입니다 " * 5}]
    msgs.append({"role": "user", "content": "마지막 질문"})
    ids = fit_messages(real_tok, msgs, 200)
    text = real_tok.decode(ids)
    assert len(ids) <= 200
    assert text.startswith("<|im_start|>system\n너는 친절한 비서다.")
    assert text.endswith("마지막 질문<|im_end|>\n<|im_start|>assistant\n")
    assert "0번째 질문" not in text and "49번째 답" in text


def test_fit_messages_truncates_single_huge_message(real_tok):
    msgs = [{"role": "user", "content": "가나다라마바사 " * 500 + "끝"}]
    ids = fit_messages(real_tok, msgs, 64)
    assert len(ids) <= 64 and real_tok.decode(ids).endswith("끝<|im_end|>\n<|im_start|>assistant\n")


def test_fit_messages_huge_system_prompt_keeps_question_and_template(real_tok):
    """RAG식 클라이언트처럼 문서를 system에 넣어 system만으로 예산을 넘는 경우. 예전에는 질문이
    빈 문자열이 되고 <|im_start|>system 머리까지 잘렸다."""
    doc = "대한민국 헌법 제1조 대한민국은 민주공화국이다. 주권은 국민에게 있다. " * 150
    msgs = [{"role": "system", "content": "참고 문서: " + doc},
            {"role": "user", "content": "위 문서에서 주권은 누구에게 있다고 하나요?"}]
    for budget in (64, 300, 1792):
        ids = fit_messages(real_tok, msgs, budget)
        text = real_tok.decode(ids)
        assert len(ids) <= budget and "�" not in text
        assert text.startswith("<|im_start|>system\n참고 문서")
        assert text.endswith("<|im_start|>user\n위 문서에서 주권은 누구에게 있다고 하나요?<|im_end|>\n"
                             "<|im_start|>assistant\n")


def test_fit_messages_cuts_only_at_user_turns_and_never_breaks_characters(real_tok):
    msgs = []
    for i in range(6):
        msgs += [{"role": "user", "content": f"질문{i} " + "가나다라 " * 30},
                 {"role": "assistant", "content": f"답{i} " + "마바사아 " * 30}]
    msgs.append({"role": "user", "content": "마지막 질문"})
    for budget in range(40, 900, 10):
        text = real_tok.decode(fit_messages(real_tok, msgs, budget))
        assert text.startswith("<|im_start|>user\n")  # 질문이 빠진 assistant 턴으로 시작하지 않는다
    # 바이트 단위 토큰을 글자 중간에서 자르면 �가 생겼다 (이모지 · 드문 음절)
    rare = "".join(chr(0xAC00 + (i * 7919) % 11172) for i in range(3000))
    for content in ("🎉🥳👍" * 400 + "끝", rare + "끝"):
        for budget in range(30, 130, 7):
            ids = fit_messages(real_tok, [{"role": "user", "content": content}], budget)
            text = real_tok.decode(ids)
            assert len(ids) <= budget and "�" not in text and text.startswith("<|im_start|>user\n")
            assert text.endswith("끝<|im_end|>\n<|im_start|>assistant\n")


def test_fit_messages_keeps_the_cut_stable_for_prefix_cache(real_tok):
    """창이 찬 뒤에도 자르는 자리가 몇 턴씩 유지돼야 이전 턴의 K·V를 재사용할 수 있다.
    예전에는 한 턴씩 밀어내 매 턴 system 뒤부터 창 전체를 다시 계산했다."""
    import random

    from model import KVCache

    rng = random.Random(0)
    words = "대한민국 헌법 주권은 국민에게 있고 서울 인공지능 데이터 학습 모델 토큰 문맥 캐시 질문".split()

    def sent(n):
        return " ".join(rng.choice(words) for _ in range(n)) + "."

    budget, cache = 1792, KVCache(1, 2048)
    msgs = [{"role": "system", "content": "너는 친절한 비서다."}]
    reused = recomputed = 0
    for turn in range(60):
        msgs.append({"role": "user", "content": sent(rng.randint(8, 25))})
        ids = fit_messages(real_tok, msgs, budget)
        k = cache.reuse(ids)
        reply = sent(rng.randint(30, 70))
        cache.tokens = ids + real_tok.encode(reply)
        cache.pos = len(cache.tokens)
        msgs.append({"role": "assistant", "content": reply})
        if turn >= 25:  # 창이 다 찬 뒤
            reused, recomputed = reused + k, recomputed + len(ids) - k
    assert reused > 3 * recomputed  # 예전 구현은 재사용 2%대


def test_fit_messages_rejects_impossible_budget(real_tok):
    import pytest

    with pytest.raises(ValueError):
        fit_messages(real_tok, [{"role": "user", "content": "안녕"}], 0)
    with pytest.raises(ValueError):  # 템플릿조차 안 들어가는 예산
        fit_messages(real_tok, [{"role": "system", "content": "긴 시스템"}, {"role": "user", "content": "안녕"}], 3)


def test_stream_decoder_never_emits_broken_syllables(real_tok):
    text = "대한민국은 동아시아의 한반도 남부에 있는 나라이다 🎉"
    ids = real_tok.encode(text)
    dec = StreamDecoder(real_tok)
    pieces = [dec.push(t) for t in ids] + [dec.flush()]
    assert "".join(pieces) == text and all("�" not in p for p in pieces)
    # 바이트 단위 토큰(음절 쪼개짐)도 처리
    raw = list(text.encode("utf-8"))
    dec = StreamDecoder(real_tok)
    assert "".join(dec.push(b) for b in raw) + dec.flush() == text
    assert dec.push(real_tok.special_tokens["<|im_end|>"]) == ""
