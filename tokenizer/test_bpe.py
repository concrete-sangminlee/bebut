"""BPE 토크나이저 검증: 왕복 일치, merge 동작, special token 처리.

실행: python -m tokenizer.test_bpe
"""

from .bpe import ByteBPETokenizer
from .pretokenize import pretokenize


def main():
    corpus = [
        "안녕하세요. 저는 서울대학교 학생입니다.",
        "안녕하세요. 오늘 날씨가 정말 좋네요.",
        "저는 오늘 학교에 갑니다. 학교에서 공부합니다.",
        "대학교에서 인공지능을 공부합니다.",
        "The quick brown fox jumps over the lazy dog. It's a test 12345.",
    ] * 50

    tok = ByteBPETokenizer.train(corpus, vocab_size=600, verbose_every=0)
    # 코퍼스가 작으면 merge할 쌍이 소진되어 목표 vocab에 못 미칠 수 있다
    assert 256 < tok.vocab_size <= 600, tok.vocab_size

    # 1) 왕복 일치 — 학습 코퍼스 밖 문자(이모지, 한자, 일본어)도 byte-level이라 무손실
    tests = [
        "안녕하세요, 세상! 저는 학교에 갑니다.",
        "BPE 토크나이저 테스트 🎉 漢字 かな 123,456원",
        "  공백   과\n줄바꿈\t탭 처리",
        "",
    ]
    for t in tests:
        assert tok.decode(tok.encode(t)) == t, t

    # 2) 자주 나온 한국어 어절이 실제로 merge되어 짧아졌는지
    ids = tok.encode("안녕하세요")
    assert len(ids) < len("안녕하세요".encode("utf-8")), ids

    # 3) special token
    s = "<|im_start|>user\n안녕<|im_end|>"
    ids = tok.encode(s, allowed_special=True)
    assert tok.special_tokens["<|im_start|>"] in ids
    assert tok.decode(ids) == s
    # allowed_special=False면 일반 텍스트로 취급 (special id가 나오면 안 됨)
    ids_plain = tok.encode(s, allowed_special=False)
    assert tok.special_tokens["<|im_start|>"] not in ids_plain
    assert tok.decode(ids_plain) == s

    # 4) 저장/로드 후 동일 결과
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tok.json")
        tok.save(p)
        tok2 = ByteBPETokenizer.load(p)
        for t in tests:
            assert tok2.encode(t) == tok.encode(t)

    # 5) 짝 없는 서로게이트는 예외 대신 U+FFFD로 (정상 텍스트는 그대로)
    assert tok.decode(tok.encode("a\ud83db")) == "a\ufffdb"
    assert tok.encode("a\ud83db") == tok.encode("a\ufffdb")

    # 6) pretokenize 동작 확인
    assert pretokenize("저는 학교에 갑니다") == ["저는", " 학교에", " 갑니다"]

    comp = len("안녕하세요. 저는 서울대학교 학생입니다.") / len(tok.encode("안녕하세요. 저는 서울대학교 학생입니다."))
    print(f"OK — vocab 600 기준 압축률: {comp:.2f} chars/token")


if __name__ == "__main__":
    main()
