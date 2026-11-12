"""OpenAI 호환 API 서버 + 웹 채팅 UI — 표준 라이브러리만 쓴다.

OpenAI 형식을 따르는 이유: openai 파이썬/JS SDK, LangChain, Open WebUI 같은 기존 도구가
base_url만 바꾸면 그대로 붙는다. 프레임워크(FastAPI 등) 없이 http.server로 만든 건
이 저장소의 원칙(무거운 의존성 없음) 때문이다. 모델은 한 번에 요청 하나만 처리한다
(GPU 한 장·배치 1 기준, 동시 요청은 순서대로 기다린다).

엔드포인트:
  GET  /                     웹 채팅 UI (web/chat.html)
  GET  /health               상태
  GET  /v1/models            모델 목록
  POST /v1/chat/completions  대화 (stream: true면 Server-Sent Events)
  POST /v1/completions       문장 이어쓰기 (베이스 모델용)

사용:
  python serve.py --ckpt checkpoints/sft_1b/latest.pt            # http://127.0.0.1:8000
  curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
       -d '{"messages":[{"role":"user","content":"안녕?"}],"stream":true}'

  from openai import OpenAI
  client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
"""

import argparse
import json
import math
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from chat import ROLES, StreamDecoder, fit_messages, nfc
from sample import load_model, pick_device

UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "chat.html")
MAX_BODY = 4 << 20  # 4MB — 이보다 큰 요청은 거절


class BadRequest(Exception):
    pass


class Engine:
    """모델 + 토크나이저 + 직렬화 락. HTTP와 무관한 생성 로직을 담아 테스트하기 쉽게 한다."""

    def __init__(self, model, tok, name="bebut", prefix_cache=True):
        self.model, self.tok, self.name = model, tok, name
        self.lock = threading.Lock()
        # 요청 사이에 K·V를 재사용한다. 같은 대화를 이어가는 요청은 기록 전체를 다시 넣지만
        # 앞부분이 지난 요청과 같으므로 그 부분은 계산하지 않는다. 다른 대화가 끼어들면
        # 공통 접두부가 짧아져 그만큼 다시 계산할 뿐이다 (락 안에서만 쓰므로 스레드 안전).
        # fp32에선 결과가 캐시 없이 생성한 것과 같다. bf16(CUDA·bf16 지원 CPU의 기본)에선 반올림 차이로
        # 같은 요청·같은 seed라도 직전 요청에 따라 드물게 다른 답이 나올 수 있어, 완전한
        # 재현이 필요하면 --no-prefix-cache로 끈다.
        self.cache = model.new_cache() if prefix_cache else None
        self.device = model.tok_emb.weight.device
        self.stop_ids = [tok.special_tokens["<|im_end|>"], tok.special_tokens["<|eos|>"]]

    # ---------------- 요청 해석 ----------------

    def sampling(self, req):
        def num(key, default, lo, hi, cast=float):
            v = req.get(key, default)
            if v is None:
                v = default
            try:
                if isinstance(v, bool) or not math.isfinite(float(v)):  # 1e400 → inf
                    raise ValueError
                v = cast(v)
            except (TypeError, ValueError, OverflowError):
                raise BadRequest(f"{key}는 유한한 숫자여야 합니다") from None
            if not lo <= v <= hi:
                raise BadRequest(f"{key}={v}는 [{lo}, {hi}] 범위를 벗어납니다")
            return v

        L = self.model.cfg.max_seq_len
        stop = req.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        if not (isinstance(stop, list) and all(isinstance(s, str) and s for s in stop)):
            raise BadRequest("stop은 문자열 또는 문자열 목록이어야 합니다")
        # OpenAI는 max_tokens 대신 max_completion_tokens를 쓰라고 하고, 새 클라이언트는 그것만 보낸다
        mt_key = "max_completion_tokens" if req.get("max_completion_tokens") is not None else "max_tokens"
        temperature = num("temperature", 0.7, 0.0, 5.0)
        return dict(
            # 문맥 길이보다 큰 값은 거절하지 않고 맞춘다 — 실제 생성 길이는 남은 문맥이 정한다
            max_tokens=min(L, num(mt_key, 256, 1, 1 << 20, int)),
            # 아주 작은 온도(1e-40 등)는 logits / T가 inf가 되어 샘플링이 죽는다 — greedy로 본다
            temperature=0.0 if temperature < 1e-4 else temperature,
            top_p=num("top_p", 0.95, 0.0, 1.0),
            top_k=num("top_k", 0, 0, self.model.cfg.vocab_size, int),
            min_p=num("min_p", 0.0, 0.0, 1.0),
            repetition_penalty=num("repetition_penalty", 1.0, 0.5, 3.0),
            seed=None if req.get("seed") is None else num("seed", 0, 0, 2**63 - 1, int),
            stop=stop[:4],
        )

    def chat_prompt(self, req, max_tokens):
        msgs = req.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise BadRequest("messages는 비어 있지 않은 목록이어야 합니다")
        clean = []
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") not in ROLES:
                raise BadRequest(f"role은 {ROLES} 중 하나여야 합니다")
            content = m.get("content", "")
            if isinstance(content, list):  # OpenAI의 [{type: text, text: ...}] 형식
                texts = [p.get("text", "") for p in content if isinstance(p, dict)]
                if not all(isinstance(t, str) for t in texts):
                    raise BadRequest("content의 text는 문자열이어야 합니다")
                content = "".join(texts)
            if not isinstance(content, str):
                raise BadRequest("content는 문자열이어야 합니다")
            clean.append({"role": m["role"], "content": content})
        L = self.model.cfg.max_seq_len
        return fit_messages(self.tok, clean, max(L // 2, L - max_tokens))

    def text_prompt(self, req, max_tokens):
        """텍스트 완성 프롬프트. 생성할 자리를 chat과 같은 규칙으로 남긴다 — 예전엔 늘 L-1토큰을
        남겨, 긴 프롬프트면 max_tokens와 무관하게 한 토큰만 생성했다."""
        prompt = req.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt:
            raise BadRequest("prompt는 비어 있지 않은 문자열이어야 합니다")
        L = self.model.cfg.max_seq_len
        return self.tok.encode(nfc(prompt))[-max(L // 2, L - max_tokens):]

    # ---------------- 생성 ----------------

    def stream(self, ids, p, chat=True):
        """텍스트 조각을 yield 하고, 끝에 (finish_reason, 생성 토큰 수, 캐시 재사용 토큰 수)를 반환한다.

        stop 문자열은 토큰 경계와 무관하게 나타나므로, 가장 긴 stop 길이-1 글자만큼은
        출력하지 않고 들고 있다가 stop이 아님이 확인되면 내보낸다.
        """
        stop_ids = self.stop_ids if chat else [self.tok.special_tokens["<|eos|>"]]
        gen = None
        if p["seed"] is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(p["seed"]))
        hold = max((len(s) for s in p["stop"]), default=1) - 1
        dec = StreamDecoder(self.tok)
        buf, n, finish = "", 0, "length"
        with self.lock:
            for t in self.model.generate_stream(
                    torch.tensor([ids], device=self.device), p["max_tokens"], p["temperature"],
                    p["top_p"], top_k=p["top_k"], min_p=p["min_p"],
                    repetition_penalty=p["repetition_penalty"], stop_ids=stop_ids, generator=gen,
                    cache=self.cache):
                n += 1
                tid = int(t)
                if tid in stop_ids:
                    finish = "stop"
                    break
                buf += dec.push(tid)
                hit = min((buf.find(s) for s in p["stop"] if s in buf), default=-1)
                if hit >= 0:
                    buf, finish = buf[:hit], "stop"
                    break
                if len(buf) > hold:
                    yield buf[: len(buf) - hold]
                    buf = buf[len(buf) - hold:]
            # 재사용한 프롬프트 토큰 수는 락 안에서 읽는다 (다음 요청이 캐시를 바꾸기 전에)
            cached = self.cache.reused if self.cache is not None else 0
        if finish == "length":
            buf += dec.flush()
        if buf:
            yield buf
        return finish, n, cached

    def complete(self, ids, p, chat=True):
        it = self.stream(ids, p, chat)
        pieces = []
        while True:
            try:
                pieces.append(next(it))
            except StopIteration as e:
                return "".join(pieces), *e.value


def usage(prompt_tokens, completion_tokens, cached_tokens):
    """OpenAI 형식 usage. cached_tokens = 프리픽스 캐시로 다시 계산하지 않은 프롬프트 토큰."""
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens}}


def make_handler(engine: Engine, verbose=False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "bebut"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if verbose:
                super().log_message(fmt, *args)

        # ---------------- 응답 도우미 ----------------

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, msg, etype="invalid_request_error"):
            self._json(code, {"error": {"message": msg, "type": etype, "code": None}})

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            # 브라우저가 요청한 헤더를 그대로 허락한다 — openai JS SDK는 X-Stainless-* 헤더도 보낸다
            asked = self.headers.get("Access-Control-Request-Headers")
            self.send_header("Access-Control-Allow-Headers", asked or "Content-Type, Authorization")

        def _sse_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.close_connection = True

        def _sse(self, obj):
            data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        # ---------------- 라우팅 ----------------

        def do_OPTIONS(self):  # CORS preflight — 브라우저의 다른 출처 페이지에서 부를 때
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                try:
                    body = open(UI_PATH, "rb").read()
                except OSError:
                    return self._error(404, "web/chat.html이 없습니다")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/health":
                self._json(200, {"status": "ok", "model": engine.name})
            elif self.path == "/v1/models":
                cfg = engine.model.cfg
                self._json(200, {"object": "list", "data": [{
                    "id": engine.name, "object": "model", "created": 0, "owned_by": "bebut",
                    "context_length": cfg.max_seq_len}]})
            else:
                self._error(404, f"없는 경로: {self.path}")

        def do_POST(self):
            # 본문은 경로와 무관하게 먼저 읽는다. 안 읽고 응답하면 keep-alive 연결에 남은
            # 본문 바이트가 다음 요청의 첫 줄로 해석된다.
            if self.headers.get("Transfer-Encoding"):
                # chunked 본문은 지원하지 않는다. 예전엔 Content-Length 0으로 보고 응답한 뒤, 남은 청크를
                # 같은 연결의 다음 요청으로 읽어 응답이 두 개 나갔다
                self.close_connection = True
                return self._error(411, "Transfer-Encoding은 지원하지 않습니다 — Content-Length를 보내세요")
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                self.close_connection = True
                return self._error(413 if n > MAX_BODY else 400, "Content-Length가 잘못됐거나 너무 큽니다")
            raw = self.rfile.read(n)
            routes = {"/v1/chat/completions": True, "/v1/completions": False}
            if self.path not in routes:
                return self._error(404, f"없는 경로: {self.path}")
            chat = routes[self.path]
            try:
                req = json.loads(raw or b"{}")
                if not isinstance(req, dict):
                    raise BadRequest("JSON 객체를 보내야 합니다")
                if not isinstance(req.get("stream_options") or {}, dict):
                    raise BadRequest("stream_options는 객체여야 합니다")
                p = engine.sampling(req)
                ids = (engine.chat_prompt if chat else engine.text_prompt)(req, p["max_tokens"])
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._error(400, "UTF-8 JSON 형식이 아닙니다")
            except (BadRequest, ValueError) as e:  # ValueError: 프롬프트를 만들 수 없는 요청 (예: 예산 초과)
                return self._error(400, str(e))

            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
            created = int(time.time())
            obj = "chat.completion" if chat else "text_completion"
            base = {"id": rid, "created": created, "model": engine.name}

            if req.get("stream"):
                self._sse_start()
                it = engine.stream(ids, p, chat)
                try:
                    if chat:  # OpenAI는 첫 청크에 role을 보낸다
                        self._sse({**base, "object": "chat.completion.chunk", "choices": [
                            {"index": 0, "delta": {"role": "assistant", "content": ""},
                             "finish_reason": None}]})
                    while True:
                        try:
                            piece = next(it)
                        except StopIteration as e:
                            finish, n_out, cached = e.value
                            break
                        choice = ({"index": 0, "delta": {"content": piece}, "finish_reason": None}
                                  if chat else {"index": 0, "text": piece, "finish_reason": None})
                        self._sse({**base, "object": obj + (".chunk" if chat else ""),
                                   "choices": [choice]})
                    last = ({"index": 0, "delta": {}, "finish_reason": finish} if chat
                            else {"index": 0, "text": "", "finish_reason": finish})
                    self._sse({**base, "object": obj + (".chunk" if chat else ""), "choices": [last]})
                    if (req.get("stream_options") or {}).get("include_usage"):
                        # OpenAI와 같이: choices가 빈 마지막 청크에 usage를 싣는다
                        self._sse({**base, "object": obj + (".chunk" if chat else ""), "choices": [],
                                   "usage": usage(len(ids), n_out, cached)})
                    self._sse("[DONE]")
                except (BrokenPipeError, ConnectionResetError):
                    it.close()  # 클라이언트가 끊으면 생성을 멈추고 락을 푼다
                except Exception as e:  # 헤더(200)는 이미 나갔다 — 오류 이벤트와 [DONE]으로 알린다
                    it.close()
                    print(f"생성 중 오류: {type(e).__name__}: {e}", flush=True)
                    try:
                        self._sse({"error": {"message": f"생성 중 오류: {type(e).__name__}",
                                             "type": "server_error", "code": None}})
                        self._sse("[DONE]")
                    except OSError:
                        pass
                return

            try:
                text, finish, n_out, cached = engine.complete(ids, p, chat)
            except Exception as e:  # 예전엔 핸들러가 죽어 응답 없이 연결이 끊겼다
                print(f"생성 중 오류: {type(e).__name__}: {e}", flush=True)
                return self._error(500, f"생성 중 오류: {type(e).__name__}", "server_error")
            choice = ({"index": 0, "message": {"role": "assistant", "content": text},
                       "finish_reason": finish} if chat
                      else {"index": 0, "text": text, "finish_reason": finish})
            self._json(200, {**base, "object": obj, "choices": [choice],
                             "usage": usage(len(ids), n_out, cached)})

    return Handler


def make_server(engine, host="127.0.0.1", port=8000, verbose=False):
    return ThreadingHTTPServer((host, port), make_handler(engine, verbose))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--host", default="127.0.0.1",
                    help="외부에 열려면 0.0.0.0 (인증이 없으니 신뢰하는 네트워크에서만)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--name", default=None, help="API에 노출할 모델 이름")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["auto", "fp32", "bf16"], default="auto")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="요청 사이 KV 캐시 재사용 끄기 (bf16에서 seed 재현이 꼭 필요할 때)")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = {"auto": None, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    model, tok = load_model(args.ckpt, args.tokenizer, device, dtype)
    name = args.name or f"bebut-{model.meta.get('config', 'model')}"
    srv = make_server(Engine(model, tok, name, prefix_cache=not args.no_prefix_cache),
                      args.host, args.port, args.verbose)
    print(f"{name} 서빙 중 — http://{args.host}:{args.port}  (device {device}, "
          f"{next(model.parameters()).dtype}) | Ctrl+C 종료", flush=True)
    if args.host not in ("127.0.0.1", "localhost"):
        print("경고: 인증 없는 서버를 외부에 열었습니다", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
