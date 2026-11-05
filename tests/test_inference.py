"""추론 경로 검증: KV 캐시, 청크 프리필, 종료 처리, 길이 한계, 샘플링 옵션."""

import pytest
import torch

from model import KVCache, apply_repetition_penalty, sample_next
from tests.conftest import make_model


def full_forward_greedy(model, prompt, n):
    out = prompt.clone()
    with torch.no_grad():
        for _ in range(n):
            logits, _ = model(out)
            out = torch.cat([out, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return out


def test_cache_generation_matches_full_forward(tiny_model):
    prompt = torch.randint(0, 1000, (2, 9))
    ref = full_forward_greedy(tiny_model, prompt, 20)
    assert torch.equal(tiny_model.generate(prompt, 20, temperature=0.0), ref)
    # 교육용 explicit 경로도 같은 결과
    assert torch.equal(tiny_model.generate(prompt, 20, temperature=0.0, explicit=True), ref)


def test_chunked_prefill_matches_full_forward(tiny_model):
    """캐시 뒤에 여러 토큰을 이어 넣는 경로(오프셋 causal 마스크)가 전체 forward와 같다."""
    idx = torch.randint(0, 1000, (2, 30))
    with torch.no_grad():
        ref, _ = tiny_model(idx)
        for explicit in (False, True):
            cache = KVCache(len(tiny_model.blocks), 30)
            tiny_model._step(idx[:, :7], cache, explicit)
            tiny_model._step(idx[:, 7:19], cache, explicit)
            last = tiny_model._step(idx[:, 19:], cache, explicit)
            assert (last - ref[:, -1]).abs().max() < 1e-4


def test_stop_ids_and_finished_rows(tiny_model):
    prompt = torch.randint(0, 1000, (2, 5))
    free = tiny_model.generate(prompt, 12, temperature=0.0)[:, 5:]
    stop = int(free[0, 3])  # 0번 행이 4번째에 내는 토큰을 종료 토큰으로
    out = tiny_model.generate(prompt, 12, temperature=0.0, eos_id=stop)[:, 5:]
    first = (out[0] == stop).nonzero()[0].item()
    assert first <= 3
    # 끝난 행은 이후 종료 토큰으로 채워진다
    assert (out[0, first:] == stop).all()
    # 모든 행이 끝나면 멈춘다 → 나머지 행도 종료 토큰을 냈거나 길이 한도
    assert (out[:, -1] == stop).all() or out.shape[1] == 12


def test_multiple_stop_ids(tiny_model):
    prompt = torch.randint(0, 1000, (1, 5))
    free = tiny_model.generate(prompt, 10, temperature=0.0)[0, 5:].tolist()
    out = tiny_model.generate(prompt, 10, temperature=0.0, eos_id=[99999, free[2]])
    assert out.shape[1] <= 5 + 3


def test_respects_max_seq_len():
    model = make_model(max_seq_len=32)
    prompt = torch.randint(0, 1000, (1, 28))
    out = model.generate(prompt, 50, temperature=0.0)
    assert out.shape[1] == 32  # 넘치지 않고 한도에서 멈춘다 (예전엔 shape 오류)
    with pytest.raises(ValueError):
        model.generate(torch.randint(0, 1000, (1, 33)), 5)


def test_topk1_and_minp1_equal_greedy():
    torch.manual_seed(0)
    logits = torch.randn(4, 500)
    greedy = logits.argmax(-1, keepdim=True)
    assert torch.equal(sample_next(logits, 1.0, top_k=1), greedy)
    assert torch.equal(sample_next(logits, 1.0, min_p=1.0), greedy)
    assert torch.equal(sample_next(logits, 1.0, top_p=1e-6), greedy)


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.randn(1, 200)
    allowed = set(torch.topk(logits, 5).indices[0].tolist())
    g = torch.Generator().manual_seed(1)
    picks = {int(sample_next(logits, 2.0, top_k=5, generator=g)) for _ in range(300)}
    assert picks <= allowed and len(picks) > 1


def test_repetition_penalty_definition():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    seen = torch.tensor([[True, True, False]])
    out = apply_repetition_penalty(logits, seen, 2.0)
    assert torch.allclose(out, torch.tensor([[1.0, -4.0, 1.0]]))


def test_strong_repetition_penalty_prevents_repeats(tiny_model):
    prompt = torch.tensor([[1, 2, 3]])
    out = tiny_model.generate(prompt, 40, temperature=0.0, repetition_penalty=1e9)[0].tolist()
    assert len(out) == len(set(out))


def test_generator_makes_sampling_reproducible(tiny_model):
    prompt = torch.randint(0, 1000, (1, 4))
    a = tiny_model.generate(prompt, 15, 1.0, 0.9, generator=torch.Generator().manual_seed(7))
    b = tiny_model.generate(prompt, 15, 1.0, 0.9, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_stream_equals_generate(tiny_model):
    prompt = torch.randint(0, 1000, (1, 6))
    streamed = [int(t) for t in tiny_model.generate_stream(prompt, 10, temperature=0.0)]
    assert streamed == tiny_model.generate(prompt, 10, temperature=0.0)[0, 6:].tolist()


def test_bf16_cast_keeps_rope_fp32(tiny_model):
    ref = tiny_model.rope.clone()
    tiny_model.to(torch.bfloat16)
    assert tiny_model.rope.dtype == torch.float32
    assert torch.equal(tiny_model.rope, ref)
    # bf16 가중치로도 생성이 돈다
    tiny_model.generate(torch.randint(0, 1000, (1, 4)), 4, temperature=0.0)


def test_min_p_renormalizes_before_top_p():
    """min-p로 후보를 줄인 뒤 top-p 누적합은 남은 후보 기준이어야 한다."""
    logits = torch.tensor([[0.5, 0.3, 0.1, 0.1]]).log()
    g = torch.Generator().manual_seed(0)
    # min_p=.3 → {0, 1}만 남고 재정규화하면 [.625, .375] → top_p=.6이면 0번만
    picks = {int(sample_next(logits, 1.0, top_p=0.6, min_p=0.3, generator=g)) for _ in range(200)}
    assert picks == {0}


def test_repetition_penalty_exempts_stop_tokens(tiny_model):
    """프롬프트에 종료 토큰이 있어도 그 logit은 깎지 않는다 (대화 프롬프트엔 늘 <|im_end|>가 있다)."""
    g = torch.Generator().manual_seed(0)
    for _ in range(2000):  # greedy 1순위가 이미 프롬프트에 있는 경우를 찾는다
        prompt = torch.randint(0, 1000, (1, 60), generator=g)
        with torch.no_grad():
            top = int(tiny_model(prompt)[0][0, -1].argmax())
        if top in prompt[0].tolist():
            break
    else:
        raise AssertionError("조건에 맞는 프롬프트를 못 찾았다")
    first = lambda **kw: int(tiny_model.generate(prompt, 1, temperature=0.0, **kw)[0, -1])  # noqa: E731
    # 종료 토큰이 아니면 벌점 때문에 1순위에서 밀려난다
    assert first(repetition_penalty=1e9) != top
    # 종료 토큰이면 벌점에서 빠져 그대로 1순위
    assert first(repetition_penalty=1e9, eos_id=top) == top


def test_prefix_cache_reuse_matches_fresh_generation(tiny_model):
    """멀티턴: 캐시를 재사용해도 매 턴 처음부터 prefill한 것과 같은 토큰이 나온다."""
    g = torch.Generator().manual_seed(0)
    cache = tiny_model.new_cache()
    hist = torch.randint(0, 1000, (1, 10), generator=g)
    prev_len = 0
    for turn in range(4):
        fresh = tiny_model.generate(hist, 8, temperature=0.0)
        reused = tiny_model.generate(hist, 8, temperature=0.0, cache=cache)
        assert torch.equal(fresh, reused), turn
        if turn:  # 지난 턴의 프롬프트+생성(마지막 토큰 제외)이 그대로 재사용된다
            assert cache.reused == prev_len - 1
        prev_len = reused.shape[1]
        hist = torch.cat([reused, torch.randint(0, 1000, (1, 5), generator=g)], dim=1)


def test_prefix_cache_handles_divergence_and_repeats(tiny_model):
    cache = tiny_model.new_cache()
    a = torch.randint(0, 1000, (1, 20))
    tiny_model.generate(a, 4, temperature=0.0, cache=cache)
    b = a.clone()
    b[0, 12] = (b[0, 12] + 1) % 1000  # 중간이 달라진 프롬프트 (오래된 턴을 잘라낸 경우 등)
    assert torch.equal(tiny_model.generate(b, 6, temperature=0.0, cache=cache),
                       tiny_model.generate(b, 6, temperature=0.0))
    assert cache.reused == 12
    # 같은 프롬프트를 다시 넣어도 (마지막 토큰은 다시 계산해야 하므로) 결과가 같다
    assert torch.equal(tiny_model.generate(b, 6, temperature=0.0, cache=cache),
                       tiny_model.generate(b, 6, temperature=0.0))
    assert cache.reused == 19
    # explicit 경로도 같은 결과
    assert torch.equal(tiny_model.generate(b, 6, temperature=0.0, cache=cache, explicit=True),
                       tiny_model.generate(b, 6, temperature=0.0))


def test_prefix_cache_rejects_batches_and_small_caches(tiny_model):
    with pytest.raises(ValueError):
        tiny_model.generate(torch.randint(0, 1000, (2, 5)), 3, cache=tiny_model.new_cache())
    with pytest.raises(ValueError):
        tiny_model.generate(torch.randint(0, 1000, (1, 5)), 30, cache=tiny_model.new_cache(16))
