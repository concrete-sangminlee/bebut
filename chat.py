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
import unicodedata

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
ROLES = ("system", "user", "assistant")


def render(messages, add_generation_prompt=True) -> str:
    """문자열 형태 (HF chat_template과 같은 결과). 표시·디버깅용."""
    s = "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    return s + (f"{IM_START}assistant\n" if add_generation_prompt else "")


def nfc(text: str) -> str:
    """NFC 정규화. 맥 파일 이름·일부 PDF 추출기가 내놓는 NFD 한글(자모 분리)은 학습 데이터에
    없던 형태라 같은 문장이 15배 넘는 바이트 조각 토큰이 된다(9 → 151토큰). 토크나이저는
    바이트 단위로 정확해야 하므로(HF·GGUF와 같은 결과) 정규화는 입력 경계에서 한다."""
    return unicodedata.normalize("NFC", text)


def _encode_turn(tok, m, trailing_nl=True):
    """메시지 하나 → (토큰 id, loss mask). 메시지는 앞뒤 메시지와 따로 토큰화된다 — 그래서
    대화 전체의 토큰 수는 턴별 토큰 수의 합이고, fit_messages가 턴을 한 번씩만 인코딩하면 된다."""
    s, e = tok.special_tokens[IM_START], tok.special_tokens[IM_END]
    role, content = m["role"], nfc(m["content"])
    if role not in ROLES:
        raise ValueError(f"알 수 없는 role: {role!r}")
    if role == "assistant":
        head, body = [s] + tok.encode("assistant\n"), tok.encode(content) + [e]
    else:
        head, body = [s] + tok.encode(f"{role}\n{content}") + [e], []
    nl = tok.encode("\n") if trailing_nl else []
    return head + body + nl, [0] * len(head) + [1] * len(body) + [0] * len(nl)


def _generation_prompt(tok):
    return [tok.special_tokens[IM_START]] + tok.encode("assistant\n")


def encode_messages(tok, messages, add_generation_prompt=True):
    """메시지 목록 → (토큰 id, loss mask). mask는 assistant 내용과 그 <|im_end|>만 1.

    add_generation_prompt=False(학습용)면 마지막 턴 뒤에 줄바꿈을 붙이지 않는다 —
    예전 SFT 예제가 <|im_end|>로 끝났던 것과 같게.
    """
    ids, mask = [], []
    for i, m in enumerate(messages):
        t_ids, t_mask = _encode_turn(tok, m, add_generation_prompt or i < len(messages) - 1)
        ids += t_ids
        mask += t_mask
    if add_generation_prompt:
        gen = _generation_prompt(tok)
        ids += gen
        mask += [0] * len(gen)
    return ids, mask


def encode_prompt(tok, messages) -> list[int]:
    """추론용: 대화 + assistant 생성 프롬프트."""
    return encode_messages(tok, messages, add_generation_prompt=True)[0]


def _clip(tok, content, n, keep="tail"):
    """content를 n토큰 이하로 — 끝(tail) 또는 앞(head)을 남긴다. 토큰은 바이트 단위라 글자
    중간에서 끊길 수 있으므로, 잘린 쪽의 반쪽 글자는 버린다 (깨진 문자 �를 만들지 않는다)."""
    if n <= 0:
        return ""
    ids = tok.encode(content)
    if len(ids) <= n:
        return content
    ids = ids[-n:] if keep == "tail" else ids[:n]
    return b"".join(tok.vocab[i] for i in ids).decode("utf-8", errors="ignore")


def fit_messages(tok, messages, budget: int, step: int | None = None) -> list[int]:
    """budget 토큰 안에 들어가도록 오래된 턴부터 버린 프롬프트.

    - 다 들어가면 그대로 둔다.
    - 넘치면 user 턴 경계에서만 자른다 — 질문이 빠진 assistant 턴으로 시작하는 프롬프트는
      SFT가 본 적 없는 형식이다. 버리는 양은 step(기본 budget//4) 토큰 단위로 올림한다:
      대화가 창을 채운 뒤에도 자르는 자리가 몇 턴 동안 그대로라 프리픽스 KV 캐시가 계속
      재사용된다 (한 턴씩 밀어내면 매 턴 창 전체를 다시 계산했다).
    - system 메시지와 마지막 메시지는 항상 남긴다. 마지막 메시지만으로도 넘치면 그 내용의
      앞부분을 잘라낸다 (질문의 끝이 보통 핵심이라서). system이 너무 길면(예: RAG 문서를
      system에 넣는 클라이언트) 마지막 메시지에 budget의 절반까지를 먼저 주고 system 내용의
      뒷부분을 잘라낸다 — 예전에는 질문이 빈 문자열이 되고 템플릿 머리까지 잘렸다.
    - 결과는 항상 <|im_start|>로 시작하고 assistant 생성 프롬프트로 끝난다.
    """
    if budget < 1:
        raise ValueError(f"budget은 1 이상이어야 한다: {budget}")
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    gen = len(_generation_prompt(tok))
    sys_len = sum(len(_encode_turn(tok, m)[0]) for m in system)
    lens = [len(_encode_turn(tok, m)[0]) for m in rest]
    total = sys_len + sum(lens) + gen
    if total <= budget:
        return encode_prompt(tok, system + rest)
    if rest:
        step = step or max(1, budget // 4)
        need = total - budget
        target = -(-need // step) * step  # 올림
        dropped, cuts = 0, []
        for k in range(1, len(rest)):
            dropped += lens[k - 1]
            if rest[k]["role"] == "user" or k == len(rest) - 1:
                cuts.append((k, dropped))
        # 올림한 만큼 버려도 창의 절반 이상이 차면 그렇게(캐시 재사용), 아니면 필요한 만큼만
        for want, floor in ((target, budget // 2), (need, 0)):
            for k, d in cuts:
                if d >= want:
                    if total - d >= floor:
                        return encode_prompt(tok, system + rest[k:])
                    break
    return _truncate(tok, system, rest[-1] if rest else None, budget)


def _truncate(tok, system, last, budget):
    """마지막 메시지(와 필요하면 system)의 내용을 잘라 budget에 맞춘다. 재인코딩으로 토큰 수가
    조금 달라질 수 있어서, 넘치면 넘친 만큼 더 줄이며 몇 번 다시 만든다 — 템플릿 머리는
    절대 잘리지 않는다."""
    def build(sys_keep, last_keep):
        msgs = []
        if system:
            # system이 여럿이면 뒤쪽 것부터 비우고, 남은 것의 앞부분을 sys_keep 토큰까지 남긴다
            left = sys_keep
            for m in system:
                n = min(left, len(tok.encode(m["content"])))
                msgs.append({**m, "content": _clip(tok, m["content"], n, keep="head")})
                left -= n
        if last is not None:
            msgs.append({**last, "content": _clip(tok, last["content"], last_keep, keep="tail")})
        return encode_prompt(tok, msgs)

    n_sys = sum(len(tok.encode(m["content"])) for m in system)
    n_last = len(tok.encode(last["content"])) if last is not None else 0
    overhead = len(build(0, 0))
    room = budget - overhead
    if room < 0:
        raise ValueError(f"budget {budget}토큰으로는 대화 템플릿({overhead}토큰)도 들어가지 않는다")
    if n_sys + n_last <= room:  # 넘친 원인이 다른 곳(턴 경계)일 때
        last_keep, sys_keep = n_last, n_sys
    else:
        last_keep = min(n_last, max(room - n_sys, room // 2))
        sys_keep = min(n_sys, room - last_keep)
    for _ in range(8):
        ids = build(sys_keep, last_keep)
        over = len(ids) - budget
        if over <= 0:
            return ids
        if sys_keep < n_sys or last_keep == 0:  # system을 이미 자르고 있으면 system에서 더 줄인다
            sys_keep = max(0, sys_keep - over)
        else:
            last_keep = max(0, last_keep - over)
    raise RuntimeError("fit_messages: budget에 맞추지 못했다")  # 여기까지 오면 버그


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
