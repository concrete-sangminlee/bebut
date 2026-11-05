"""Decoder-only 트랜스포머 — Llama 3 / Qwen3 계열 최신 구성.

구성 요소와 채택 이유:
- Pre-RMSNorm: LayerNorm에서 평균 빼기(centering)를 제거한 형태. 성능 동급에 연산 절약.
  pre-norm(블록 입구에서 정규화)이 post-norm보다 깊은 모델에서 안정적.
- RoPE: 위치를 Q·K 벡터의 회전으로 인코딩. 학습 길이 밖 일반화와 상대 위치 표현이
  learned positional embedding보다 우수. 최신 모델 전부 채택.
- GQA: K·V head를 Q head보다 적게 둬서 KV cache와 연산 절감. (n_head=16, n_kv_head=4면
  Q head 4개가 KV head 1개를 공유)
- QK-Norm: attention 직전 Q·K를 head 단위로 RMSNorm. logit 폭주를 막아 학습 안정성 개선
  (Qwen3, Gemma 계열 채택).
- SwiGLU: gate * up 곱 구조의 MLP. 같은 파라미터로 GELU MLP보다 품질 우수.
- bias 없는 Linear, embedding tying(입출력 임베딩 공유 — 소형 모델에서 파라미터 절약).

attention은 교육용 명시적 구현(explicit)과 F.scaled_dot_product_attention(SDPA,
FlashAttention 커널) 두 경로를 두고 단위테스트로 일치를 검증한다.

추론은 미리 할당한 KV 캐시(KVCache) 위에서 SDPA로 돈다. 예전 구현은 토큰마다
캐시 전체를 torch.cat으로 복사하고 (Tk × Tk) causal 마스크를 층마다 새로 만들어,
문맥이 길어질수록 느려졌다.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 65536
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    # SwiGLU hidden dim. 관례상 (8/3)*dim을 64의 배수로 올림
    ffn_dim: int | None = None

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0
        if self.ffn_dim is None:
            self.ffn_dim = ((int(self.dim * 8 / 3) + 63) // 64) * 64


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # float32로 계산해 bf16 정밀도 손실 방지 (표준 관행)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    """RoPE 회전각의 cos/sin 테이블. shape (max_seq_len, dim/2), complex 대신 실수 쌍."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)  # (T, dim/2)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (T, dim/2, 2)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, hd). 인접 차원 쌍 (x0,x1)을 각도 θ만큼 회전."""
    B, H, T, hd = x.shape
    x = x.float().view(B, H, T, hd // 2, 2)
    cos, sin = rope[:T, :, 0], rope[:T, :, 1]  # (T, hd/2)
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(3)


class KVCache:
    """층별 K·V를 담는 미리 할당된 버퍼.

    토큰마다 torch.cat으로 캐시를 이어붙이면 길이 L까지 생성하는 동안 O(L²) 바이트를
    복사한다. 최대 길이만큼 한 번 잡아두고 제자리에 써 넣으면 복사가 새 토큰분뿐이다.
    pos는 지금까지 채워진 길이 = 다음 토큰의 절대 위치(RoPE 각도)다.
    """

    def __init__(self, n_layers: int, max_len: int):
        self.n_layers = n_layers
        self.max_len = max_len
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers
        self.pos = 0
        self.tokens: list[int] = []  # 캐시에 들어 있는 토큰 (배치 1일 때만 추적)
        self.reused = 0  # 마지막 reuse()에서 재사용한 토큰 수 (진단용)

    def reuse(self, tokens: list[int]) -> int:
        """새 프롬프트와 캐시의 공통 접두부만 남기고, 재사용할 토큰 수를 돌려준다.

        대화는 턴마다 '지금까지의 기록 + 새 질문'을 통째로 다시 넣는다. 앞부분은 지난 턴에
        이미 계산한 K·V와 같으므로 그대로 두고 달라진 지점부터만 prefill하면 된다.
        마지막 토큰은 항상 다시 넣는다 — 다음 토큰의 logits가 그 위치의 출력이라서.
        캐시 뒤쪽의 낡은 값은 지우지 않는다: pos 이후는 다음 update가 덮어쓰고,
        읽을 때도 [:pos+T]까지만 본다.
        """
        k = 0
        for a, b in zip(self.tokens, tokens):
            if a != b:
                break
            k += 1
        k = min(k, len(tokens) - 1, self.pos)
        self.pos, self.tokens, self.reused = k, self.tokens[:k], k
        return k

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """새 K·V(B, Hkv, T, hd)를 기록하고, 지금까지의 전체 K·V 뷰를 돌려준다."""
        B, H, T, hd = k.shape
        if self.pos + T > self.max_len:
            raise ValueError(f"KV 캐시 초과: {self.pos} + {T} > max_len {self.max_len}")
        if self.k[layer] is None:  # dtype은 첫 기록에서 정한다 (autocast면 v만 bf16일 수 있다)
            self.k[layer] = k.new_empty(B, H, self.max_len, hd)
            self.v[layer] = v.new_empty(B, H, self.max_len, hd)
        end = self.pos + T
        self.k[layer][:, :, self.pos:end] = k
        self.v[layer][:, :, self.pos:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n: int):
        """모든 층에 n개 토큰을 기록한 뒤 호출 — 위치를 전진시킨다."""
        self.pos += n

    def reorder(self, keep: torch.Tensor):
        """배치 행을 골라낸다 (끝난 시퀀스를 빼서 남은 행만 계속 생성할 때)."""
        for i in range(self.n_layers):
            if self.k[i] is not None:
                self.k[i] = self.k[i][keep]
                self.v[i] = self.v[i][keep]


def causal_mask(T: int, Tk: int, device) -> torch.Tensor:
    """새 쿼리 T개(절대 위치 Tk-T … Tk-1)가 볼 수 있는 키 = 자기 위치 이하.

    캐시가 있으면 쿼리와 키 길이가 달라서 SDPA의 is_causal(좌상단 정렬)을 쓰면 틀린다.
    True = 볼 수 있음.
    """
    return torch.ones(T, Tk, dtype=torch.bool, device=device).tril(diagonal=Tk - T)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

    def forward(self, x, rope, cache: KVCache | None = None, layer: int = 0,
                explicit: bool = False):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)  # QK-Norm은 RoPE 이전에 적용

        # cache 사용 시 새 토큰의 절대 위치는 pos부터 시작
        pos = 0 if cache is None else cache.pos
        q = apply_rope(q, rope[pos:pos + T]).to(x.dtype)
        k = apply_rope(k, rope[pos:pos + T]).to(x.dtype)

        if cache is not None:
            k, v = cache.update(layer, k, v)
        Tk = k.shape[2]
        rep = self.n_heads // self.n_kv_heads

        if explicit:
            # 교육용 명시적 구현 — SDPA와 일치해야 함 (단위테스트로 검증)
            # GQA: KV head를 Q head 수만큼 반복
            k_r = k.repeat_interleave(rep, dim=1)
            v_r = v.repeat_interleave(rep, dim=1)
            scores = (q @ k_r.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~causal_mask(T, Tk, x.device), float("-inf"))
            out = F.softmax(scores.float(), dim=-1).to(q.dtype) @ v_r
        elif T == 1 and Tk > 1:
            # 디코드 1토큰: 같은 KV head를 공유하는 Q head rep개를 '길이 rep짜리 쿼리'로
            # 묶으면 K·V를 반복 복사하지 않고 한 번에 계산된다 (마스크도 필요 없다).
            # q head h는 KV head h // rep를 본다 — repeat_interleave와 같은 대응.
            qg = q.reshape(B, self.n_kv_heads, rep, self.head_dim)
            out = F.scaled_dot_product_attention(qg, k, v.to(qg.dtype))
            out = out.reshape(B, self.n_heads, 1, self.head_dim)
        else:
            k_r = k.repeat_interleave(rep, dim=1)
            v_r = v.repeat_interleave(rep, dim=1)
            if Tk == T:  # 학습·프리필: 표준 causal (FlashAttention 경로)
                out = F.scaled_dot_product_attention(q, k_r, v_r, is_causal=True)
            else:  # 캐시 뒤에 여러 토큰을 이어 넣는 경우 (청크 프리필)
                out = F.scaled_dot_product_attention(
                    q, k_r, v_r, attn_mask=causal_mask(T, Tk, x.device))

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def _chunk_loss(h_chunk, t_chunk, w):
    """한 청크의 logits를 만들어 loss 합을 반환. 체크포인팅으로 감싸 쓰인다."""
    logits = F.linear(h_chunk, w).float()
    return F.cross_entropy(logits, t_chunk, ignore_index=-100, reduction="sum")


def chunked_cross_entropy(h, targets, w, chunk: int):
    """vocab이 클 때 logits 전체를 메모리에 올리지 않는 cross-entropy.

    64K vocab · 32K 토큰이면 fp32 logits만 8.6GB라 그대로는 48GB GPU도 터진다.
    시퀀스를 청크로 쪼개 계산하고, gradient checkpointing으로 backward 때
    청크 하나씩만 다시 만들어 쓴다 — 순간 메모리가 1/N로 줄고 결과는 동일하다.
    """
    h = h.reshape(-1, h.size(-1))
    t = targets.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32)
    n_valid = (t != -100).sum()
    for i in range(0, h.size(0), chunk):
        hc, tc = h[i : i + chunk], t[i : i + chunk]
        if torch.is_grad_enabled() and h.requires_grad:
            total = total + torch.utils.checkpoint.checkpoint(
                _chunk_loss, hc, tc, w, use_reentrant=False
            )
        else:
            total = total + _chunk_loss(hc, tc, w)
    return total / n_valid.clamp(min=1)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, rope, cache=None, layer=0, explicit=False):
        x = x + self.attn(self.attn_norm(x), rope, cache, layer, explicit)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def apply_repetition_penalty(logits, seen: torch.Tensor, penalty: float):
    """이미 나온 토큰의 logit을 깎는다 (CTRL 논문 방식, HF와 동일한 정의).

    양수 logit은 penalty로 나누고 음수는 곱한다 — 어느 쪽이든 확률이 내려간다.
    SFT 모델의 반복 루프(평가에서 3~8%)를 재학습 없이 줄이는 추론 시점 장치다.
    """
    if penalty == 1.0:
        return logits
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalized, logits)


def sample_next(logits, temperature=0.8, top_p=1.0, top_k=0, min_p=0.0, generator=None):
    """logits (B, V) → 다음 토큰 (B, 1).

    temperature <= 0이면 greedy. 필터 순서는 temperature → top-k → min-p → top-p.
    min-p: 최고 확률의 min_p배 미만인 토큰을 버린다 — 분포가 뾰족할 땐 후보를 좁히고
    평평할 땐 넓혀서, 고정 top-p보다 온도를 올렸을 때 덜 무너진다.
    """
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if min_p > 0:
        floor = probs.max(-1, keepdim=True).values * min_p
        probs = probs.masked_fill(probs < floor, 0.0)
        # 다시 정규화해야 top-p의 누적합이 남은 후보 기준이 된다 (HF와 같은 순서·의미)
        probs = probs / probs.sum(-1, keepdim=True)
    # nucleus (top-p) 샘플링: 누적 확률이 top_p를 넘기 직전까지만 남긴다
    sorted_p, sorted_idx = probs.sort(descending=True)
    if top_p < 1.0:
        cum = sorted_p.cumsum(-1)
        sorted_p = sorted_p.masked_fill(cum - sorted_p > top_p, 0.0)
    sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
    pick = torch.multinomial(sorted_p, 1, generator=generator)
    return sorted_idx.gather(-1, pick)


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # embedding tying

        rope = precompute_rope(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope", rope, persistent=False)

        self.apply(self._init_weights)
        # residual 경로에 더해지는 projection은 깊이에 따라 스케일 다운 (GPT-2 레시피)
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("w_down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _apply(self, fn, recurse=True):
        """model.to(torch.bfloat16)이 와도 RoPE 테이블은 fp32로 다시 만든다.

        bf16의 유효숫자는 3자리 남짓이라 cos/sin을 bf16으로 반올림하면 위치 정보가
        뭉개진다. 학습은 fp32 가중치 + autocast라 문제가 없었지만, 추론에서 가중치를
        bf16으로 내리면 버퍼까지 같이 내려간다.
        """
        super()._apply(fn, recurse)
        if self.rope.dtype != torch.float32:
            cfg = self.cfg
            self.rope = precompute_rope(cfg.dim // cfg.n_heads, cfg.max_seq_len,
                                        cfg.rope_theta).to(self.rope.device)
        return self

    def hidden_states(self, idx, explicit=False):
        """최종 RMSNorm까지 통과한 hidden (B, T, dim). lm_head 직전 값."""
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope, explicit=explicit)
        return self.final_norm(x)

    def forward(self, idx, targets=None, explicit=False, loss_chunk=None):
        x = self.hidden_states(idx, explicit)
        if targets is not None:
            if loss_chunk:
                return None, chunked_cross_entropy(x, targets, self.lm_head.weight, loss_chunk)
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None  # 추론 시 마지막 위치만

    def _step(self, idx, cache: KVCache, explicit=False):
        """캐시 뒤에 idx (B, T)를 이어 넣고 마지막 위치의 logits (B, V)를 반환."""
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, cache, i, explicit)
        cache.advance(idx.shape[1])
        return self.lm_head(self.final_norm(x[:, -1]))

    def new_cache(self, max_len: int | None = None) -> KVCache:
        """턴을 넘겨 재사용할 캐시 (generate_stream(cache=...)에 넘긴다). 기본 크기는 max_seq_len."""
        return KVCache(len(self.blocks), max_len or self.cfg.max_seq_len)

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=0.8, top_p=0.95, *,
                        top_k=0, min_p=0.0, repetition_penalty=1.0, stop_ids=None,
                        generator=None, explicit=False, cache: KVCache | None = None):
        """토큰을 하나씩 내놓는 생성기. 매 step (B, 1) 텐서를 yield 한다.

        - 모든 행이 stop_ids 중 하나를 낸 뒤 멈춘다. 먼저 끝난 행은 이후 stop_ids[0]으로
          채워진다 (이전 구현은 끝난 행에도 계속 아무 토큰이나 생성했다).
        - 전체 길이는 max_seq_len을 넘지 않는다. RoPE 테이블이 거기까지라서, 넘으면
          예전 구현은 shape 오류로 죽었다.
        - cache(new_cache()로 만든 것, 배치 1)를 넘기면 지난 호출과 공통인 접두부는 다시
          계산하지 않는다. 멀티턴 대화에서 기록 전체를 매 턴 prefill하던 비용이 사라진다.
        """
        B, T = idx.shape
        L = self.cfg.max_seq_len
        if T > L:
            raise ValueError(f"프롬프트 {T}토큰이 max_seq_len {L}보다 길다 — 앞부분을 잘라서 넣을 것")
        n_new = min(max_new_tokens, L - T)
        if n_new <= 0:
            return
        if isinstance(stop_ids, int):
            stop_ids = [stop_ids]
        stop = (torch.tensor(list(stop_ids), device=idx.device)
                if stop_ids else None)
        seen = None
        if repetition_penalty != 1.0:
            seen = torch.zeros(B, self.cfg.vocab_size, dtype=torch.bool, device=idx.device)
            seen.scatter_(1, idx, True)
            # 종료 토큰은 벌점에서 뺀다. 대화 프롬프트엔 <|im_end|>가 항상 들어 있어서,
            # 빼지 않으면 첫 토큰부터 '멈추기'가 억제돼 답이 늘어진다.
            if stop is not None:
                seen[:, stop] = False
        done = torch.zeros(B, dtype=torch.bool, device=idx.device)

        start = 0
        if cache is None:
            cache = KVCache(len(self.blocks), T + n_new)
        else:
            if B != 1:
                raise ValueError("캐시 재사용은 배치 1만 지원한다")
            if cache.max_len < T + n_new:
                raise ValueError(f"캐시가 작다: {cache.max_len} < {T + n_new}")
            start = cache.reuse(idx[0].tolist())
        track = B == 1
        logits = self._step(idx[:, start:], cache, explicit)  # 프롬프트 프리필 (한 번에)
        if track:
            cache.tokens = idx[0].tolist()
        for i in range(n_new):
            if seen is not None:
                logits = apply_repetition_penalty(logits, seen, repetition_penalty)
            nxt = sample_next(logits, temperature, top_p, top_k, min_p, generator)
            if stop is not None:
                nxt = torch.where(done[:, None], stop[0], nxt)
                done |= torch.isin(nxt[:, 0], stop)
            yield nxt
            if (stop is not None and bool(done.all())) or i == n_new - 1:
                return
            if seen is not None:
                seen.scatter_(1, nxt, True)
                if stop is not None:
                    seen[:, stop] = False
            logits = self._step(nxt, cache, explicit)
            if track:
                cache.tokens.append(int(nxt))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.95, eos_id=None, **kw):
        """KV cache 기반 생성. idx: (B, T) 프롬프트 → (B, T + 생성 길이).

        eos_id는 정수 하나 또는 정수 리스트(여러 종료 토큰). 나머지 인자는
        generate_stream 참고 (top_k, min_p, repetition_penalty, generator, explicit).
        """
        toks = list(self.generate_stream(idx, max_new_tokens, temperature, top_p,
                                         stop_ids=eos_id, **kw))
        return torch.cat([idx, *toks], dim=1) if toks else idx

    def num_params(self, non_embedding=True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()  # tying이라 lm_head와 동일 텐서
        return n

    def param_groups(self, weight_decay: float):
        """2D 이상(행렬)에만 weight decay, 1D(norm 가중치)는 제외 — 표준 관행."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
