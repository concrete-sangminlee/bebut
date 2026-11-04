"""대화 템플릿 — 학습(SFT)과 추론(대화 CLI·API 서버·평가)이 공유하는 단 하나의 정의.

형식 (ChatML):
    <|im_start|>user\\n{질문}<|im_end|>\\n<|im_start|>assistant\\n{답}<|im_end|>

예전에는 이 형식이 sft.py, sample.py, sft_eval.py, prepare_anneal.py에 각각 문자열로
적혀 있었다. 한 곳만 바뀌어도 모델은 학습 때 본 적 없는 형식을 받게 되는데,
이런 어긋남은 오류 없이 품질만 조용히 떨어뜨린다. 그래서 여기 하나로 모은다.

토큰화 규칙은 SFT 학습 때와 토큰 단위로 같아야 한다:
- user/system 턴: encode("{role}\\n{내용}")을 통째로 (학습 코드가 그렇게 했다)
- assistant 턴: encode("assistant\\n")과 encode(내용)을 따로 — 추론 때 모델은
  "assistant\\n" 뒤에서부터 내용을 생성하므로 둘 사이가 토큰 경계여야 한다
- 턴 사이에는 encode("\\n")
"""

import codecs

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
ROLES = ("system", "user", "assistant")


def render(messages, add_generation_prompt=True) -> str:
    """문자열 형태 (HF chat_template과 같은 결과). 표시·디버깅용."""
    s = "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    return s + (f"{IM_START}assistant\n" if add_generation_prompt else "")


def encode_messages(tok, messages, add_generation_prompt=True):
    """메시지 목록 → (토큰 id, loss mask). mask는 assistant 내용과 그 <|im_end|>만 1.

    add_generation_prompt=False(학습용)면 마지막 턴 뒤에 줄바꿈을 붙이지 않는다 —
    예전 SFT 예제가 <|im_end|>로 끝났던 것과 같게.
    """
    s, e = tok.special_tokens[IM_START], tok.special_tokens[IM_END]
    nl = tok.encode("\n")
    ids, mask = [], []
    for i, m in enumerate(messages):
        role, content = m["role"], m["content"]
        if role not in ROLES:
            raise ValueError(f"알 수 없는 role: {role!r}")
        if role == "assistant":
            head, body = [s] + tok.encode("assistant\n"), tok.encode(content) + [e]
        else:
            head, body = [s] + tok.encode(f"{role}\n{content}") + [e], []
        ids += head + body
        mask += [0] * len(head) + [1] * len(body)
        if add_generation_prompt or i < len(messages) - 1:
            ids += nl
            mask += [0] * len(nl)
    if add_generation_prompt:
        gen = [s] + tok.encode("assistant\n")
        ids += gen
        mask += [0] * len(gen)
    return ids, mask


def encode_prompt(tok, messages) -> list[int]:
    """추론용: 대화 + assistant 생성 프롬프트."""
    return encode_messages(tok, messages, add_generation_prompt=True)[0]


def fit_messages(tok, messages, budget: int) -> list[int]:
    """budget 토큰 안에 들어가도록 오래된 턴부터 버린 프롬프트.

    system 메시지와 마지막 메시지는 항상 남긴다. 마지막 메시지 하나도 안 들어가면
    그 내용의 앞부분을 잘라낸다 (질문의 끝이 보통 핵심이라서). 예전 대화 CLI는
    기록을 무한히 붙여서 2048토큰을 넘는 순간 RoPE 테이블 밖으로 나가 죽었다.
    """
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if not rest:
        return encode_prompt(tok, system)[-budget:]
    for k in range(len(rest)):
        ids = encode_prompt(tok, system + rest[k:])
        if len(ids) <= budget:
            return ids
    last = dict(rest[-1])
    content_ids = tok.encode(last["content"])
    overhead = len(encode_prompt(tok, system + [{**last, "content": ""}]))
    keep = max(0, budget - overhead - 1)
    last["content"] = tok.decode(content_ids[-keep:]) if keep else ""
    ids = encode_prompt(tok, system + [last])
    return ids[-budget:]  # 디코드-재인코드로 한두 토큰 늘었을 때의 안전장치


def instruction_to_messages(instruction, output, inp=None, system=None):
    """KoAlpaca/KULLM식 (instruction, input, output) → messages. 예전 SFT와 같은 결합 방식."""
    user = instruction + (f"\n\n{inp}" if inp else "")
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": user}, {"role": "assistant", "content": output}]


class StreamDecoder:
    """토큰을 하나씩 받아 출력 가능한 텍스트 조각을 내놓는다.

    한글 음절은 UTF-8 3바이트이고 byte-level BPE 토큰은 음절 중간에서 끊길 수 있다.
    토큰마다 decode하면 반쪽 음절이 깨진 문자(�)로 찍힌다. 증분 디코더는 완성되지
    않은 바이트를 들고 있다가 다음 토큰과 합쳐서 내보낸다.
    """

    def __init__(self, tok):
        self.tok = tok
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        if token_id in self.tok.id_to_special:
            return ""
        return self._dec.decode(self.tok.vocab.get(token_id, b""))

    def flush(self) -> str:
        return self._dec.decode(b"", final=True)
