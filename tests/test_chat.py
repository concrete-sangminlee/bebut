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
