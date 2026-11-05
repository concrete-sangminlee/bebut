"""API 서버 검증 — 실제 HTTP 요청으로 OpenAI 호환 동작을 확인한다."""

import http.client
import json
import socket
import threading

import pytest

from serve import Engine, make_server
from tests.conftest import make_model


@pytest.fixture(scope="module")
def server(tiny_tok):
    model = make_model(vocab_size=1024, max_seq_len=256)
    srv = make_server(Engine(model, tiny_tok, "tiny"), "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def request(port, method, path, body=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    c.request(method, path, body=data, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    payload = r.read()
    c.close()
    return r.status, r.getheader("Content-Type"), payload


def post(port, path, body):
    status, ctype, payload = request(port, "POST", path, body)
    return status, (json.loads(payload) if "json" in (ctype or "") else payload.decode())


def sse_events(payload: str):
    out = []
    for block in payload.strip().split("\n\n"):
        data = block.removeprefix("data: ")
        out.append(data if data == "[DONE]" else json.loads(data))
    return out


CHAT = {"messages": [{"role": "user", "content": "안녕하세요"}], "temperature": 0, "max_tokens": 24}


def test_health_models_and_ui(server):
    assert json.loads(request(server, "GET", "/health")[2])["status"] == "ok"
    models = json.loads(request(server, "GET", "/v1/models")[2])
    assert models["data"][0]["id"] == "tiny" and models["data"][0]["context_length"] == 256
    status, ctype, body = request(server, "GET", "/")
    assert status == 200 and ctype.startswith("text/html") and b"/v1/chat/completions" in body


def test_chat_non_stream_shape(server):
    status, r = post(server, "/v1/chat/completions", CHAT)
    assert status == 200 and r["object"] == "chat.completion"
    choice = r["choices"][0]
    assert choice["message"]["role"] == "assistant" and isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "length")
    u = r["usage"]
    assert u["prompt_tokens"] > 0 and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_stream_equals_non_stream(server):
    _, full = post(server, "/v1/chat/completions", CHAT)
    status, payload = post(server, "/v1/chat/completions", {**CHAT, "stream": True})
    ev = sse_events(payload)
    assert status == 200 and ev[-1] == "[DONE]"
    assert ev[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev[:-1])
    assert text == full["choices"][0]["message"]["content"]
    assert ev[-2]["choices"][0]["finish_reason"] == full["choices"][0]["finish_reason"]


def test_seeded_sampling_is_reproducible(server):
    body = {**CHAT, "temperature": 1.0, "seed": 3}
    a = post(server, "/v1/chat/completions", body)[1]["choices"][0]["message"]["content"]
    b = post(server, "/v1/chat/completions", body)[1]["choices"][0]["message"]["content"]
    assert a == b


def test_stop_strings(server):
    base = {**CHAT, "max_tokens": 40}
    text = post(server, "/v1/chat/completions", base)[1]["choices"][0]["message"]["content"]
    if len(text) < 6:
        pytest.skip("무작위 모델의 출력이 너무 짧다")
    stop = text[3:5]
    expect = text[: text.find(stop)]
    r = post(server, "/v1/chat/completions", {**base, "stop": [stop]})[1]["choices"][0]
    assert r["message"]["content"] == expect and r["finish_reason"] == "stop"
    ev = sse_events(post(server, "/v1/chat/completions", {**base, "stop": stop, "stream": True})[1])
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in ev[:-1]) == expect


def test_completions_endpoint(server):
    status, r = post(server, "/v1/completions", {"prompt": "대한민국의", "max_tokens": 8,
                                                  "temperature": 0})
    assert status == 200 and r["object"] == "text_completion" and "text" in r["choices"][0]
    ev = sse_events(post(server, "/v1/completions", {"prompt": "대한민국의", "max_tokens": 8,
                                                     "temperature": 0, "stream": True})[1])
    assert "".join(e["choices"][0]["text"] for e in ev[:-1]) == r["choices"][0]["text"]


@pytest.mark.parametrize("body,raw", [
    (None, b"{not json"),
    ({"messages": []}, None),
    ({"messages": [{"role": "robot", "content": "x"}]}, None),
    ({**CHAT, "temperature": 99}, None),
    ({**CHAT, "max_tokens": "많이"}, None),
    ({"prompt": ""}, None),
    ({**CHAT, "seed": "abc"}, None),
    ({**CHAT, "seed": 1e30}, None),
    ({**CHAT, "seed": [1]}, None),
    ({"messages": [{"role": "user", "content": [{"type": "text", "text": 5}]}]}, None),
    (None, b'{"messages": "\xff\xfe"}'),
])
def test_bad_requests_get_400(server, body, raw):
    path = "/v1/completions" if body and "prompt" in body else "/v1/chat/completions"
    status, _, payload = request(server, "POST", path, body, raw)
    assert status == 400 and "message" in json.loads(payload)["error"]
    if body is not None:  # 스트리밍이어도 200 + 끊긴 스트림이 아니라 400이어야 한다
        assert request(server, "POST", path, {**body, "stream": True})[0] == 400


def test_unknown_path_404(server):
    assert request(server, "GET", "/nope")[0] == 404
    assert request(server, "POST", "/v1/nope", {})[0] == 404


def test_keep_alive_after_404_reads_body(server):
    """404여도 본문을 소비해야 같은 연결의 다음 요청이 정상 처리된다."""
    c = http.client.HTTPConnection("127.0.0.1", server, timeout=60)
    for path, code in (("/v1/nope", 404), ("/v1/chat/completions", 200)):
        c.request("POST", path, body=json.dumps(CHAT).encode(),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        r.read()
        assert r.status == code
    c.close()


def test_client_disconnect_releases_lock(server):
    """스트리밍 도중 클라이언트가 끊어도 다음 요청이 막히지 않아야 한다."""
    s = socket.create_connection(("127.0.0.1", server), timeout=30)
    body = json.dumps({**CHAT, "max_tokens": 200, "stream": True}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    got = b""
    while b"data:" not in got:  # 헤더와 첫 청크가 따로 올 수 있다
        chunk = s.recv(4096)
        assert chunk, "첫 SSE 청크 전에 연결이 닫혔다"
        got += chunk
    s.close()
    status, r = post(server, "/v1/chat/completions", CHAT)
    assert status == 200


def test_engine_cache_reuse_matches_fresh_engine(tiny_tok):
    """요청 사이 K·V 재사용: 같은 대화를 이어가든, 다른 대화가 끼어들든 결과가 같아야 한다."""
    model = make_model(vocab_size=1024, max_seq_len=256)
    shared = Engine(model, tiny_tok, "tiny")
    p = shared.sampling({"temperature": 0, "max_tokens": 12})

    def ask(engine, msgs):
        return engine.complete(engine.chat_prompt({"messages": msgs}, p["max_tokens"]), p)[0]

    a = [{"role": "user", "content": "안녕하세요"}]
    b = [{"role": "user", "content": "대한민국의 수도는"}]
    for turn in range(3):
        for conv in (a, b):  # 두 대화가 번갈아 들어온다
            got = ask(shared, conv)
            assert got == ask(Engine(model, tiny_tok, "fresh"), conv), (turn, conv[0]["content"])
            conv += [{"role": "assistant", "content": got}, {"role": "user", "content": f"질문 {turn}"}]
    assert shared.cache.reused > 0


def test_engine_without_prefix_cache(tiny_tok):
    model = make_model(vocab_size=1024, max_seq_len=256)
    e = Engine(model, tiny_tok, "tiny", prefix_cache=False)
    p = e.sampling({"temperature": 0, "max_tokens": 8})
    ids = e.chat_prompt({"messages": [{"role": "user", "content": "안녕하세요"}]}, 8)
    assert e.cache is None
    assert e.complete(ids, p)[0] == Engine(model, tiny_tok, "x").complete(ids, p)[0]


def test_usage_reports_cached_tokens_and_streams_usage_chunk(server):
    """같은 대화를 이어 보내면 usage.prompt_tokens_details.cached_tokens에 재사용분이 잡힌다."""
    msgs = [{"role": "user", "content": "대한민국의 수도는"}]
    first = post(server, "/v1/chat/completions", {"messages": msgs, "temperature": 0, "max_tokens": 6})[1]
    msgs += [{"role": "assistant", "content": first["choices"][0]["message"]["content"]},
             {"role": "user", "content": "또 질문"}]
    ev = sse_events(post(server, "/v1/chat/completions",
                         {"messages": msgs, "temperature": 0, "max_tokens": 6, "stream": True,
                          "stream_options": {"include_usage": True}})[1])
    assert ev[-1] == "[DONE]" and ev[-2]["choices"] == []
    u = ev[-2]["usage"]
    assert u["prompt_tokens_details"]["cached_tokens"] > first["usage"]["prompt_tokens"] // 2
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    # include_usage가 없으면 usage 청크를 보내지 않는다
    ev2 = sse_events(post(server, "/v1/chat/completions", {**CHAT, "stream": True})[1])
    assert all("usage" not in e for e in ev2[:-1])
    assert request(server, "POST", "/v1/chat/completions", {**CHAT, "stream_options": 3})[0] == 400
