# 추론 가이드 — 어디서 어떻게 돌릴까

1.2B 모델을 돌리는 방법은 세 가지입니다. 이 문서의 숫자는 모두 **1.2B 실제 크기**
(`configs/base_1b`, 무작위 가중치 — 속도는 가중치 값과 무관)로 잰 실측입니다.

측정 환경: Intel Xeon 2.1GHz **4코어**(AVX512-BF16·AMX 지원), 메모리 15GB, GPU 없음.
코어가 더 많거나 메모리 대역폭이 큰 기계에선 전부 빨라지지만 순서는 같습니다.

## 디코드 속도 (토큰 생성)

| 경로 | 형식 | 크기 | 속도 |
|---|---|---|---|
| PyTorch (`sample.py`, `serve.py`) | fp32 | 4.8GB | 5.3 tok/s |
| PyTorch | bf16 | 2.4GB | 6.9 tok/s |
| llama.cpp (`export_gguf.py`) | F16 | 2.4GB | 12.4 tok/s |
| **llama.cpp** | **Q8_0** | **1.28GB** | **17.3 tok/s** |

디코드는 토큰마다 가중치 전체를 메모리에서 읽어야 해서 **가중치 크기가 곧 속도**입니다.
그래서 bf16(절반)이 fp32보다 빠르고, 8비트 Q8_0이 가장 빠릅니다. llama.cpp는 CPU용으로
손으로 최적화한 커널이라 같은 F16끼리 비교해도 PyTorch bf16보다 1.8배 빠릅니다.

`sample.py`·`serve.py`는 CPU가 bf16을 하드웨어로 지원하면(AVX512-BF16·AMX) **자동으로 bf16**을
씁니다. 지원하지 않는 CPU에서 bf16은 소프트웨어로 흉내 내서 오히려 느리므로 fp32를 씁니다
(`--dtype fp32|bf16`으로 강제할 수 있습니다). 평가 스크립트는 점수가 dtype에 흔들리지 않게 항상 fp32입니다.

## 대화가 길어질 때 — 프리픽스 캐시

대화는 턴마다 기록 전체를 다시 넣습니다. 기록 1,000토큰 + 새 질문 60토큰일 때 첫 토큰까지:

| | 첫 토큰까지 |
|---|---|
| 캐시 없이 (매 턴 1,060토큰 전부 계산) | 2.7초 |
| **프리픽스 캐시** (지난 턴과 달라진 60토큰만) | **0.36초** |

`sample.py --chat`과 `serve.py`는 기본으로 켜져 있고, 서버 응답의
`usage.prompt_tokens_details.cached_tokens`와 웹 UI의 "문맥 N토큰 중 M토큰 캐시 재사용"으로
확인할 수 있습니다. fp32에선 결과가 캐시 없이 생성한 것과 토큰 단위로 같습니다. bf16에선
반올림 차이로 드물게 달라질 수 있어, 완전한 재현이 필요하면 `--no-prefix-cache`로 끕니다.

## 권장 구성

| 상황 | 권장 |
|---|---|
| CPU에서 대화·서빙 (학습 중이라 GPU를 비워둬야 할 때 포함) | **GGUF Q8_0 + llama.cpp / Ollama** |
| CPU에서 이 저장소 코드로 실험 (샘플링 옵션·평가와 같은 코드 경로) | `serve.py` / `sample.py` (bf16 자동) |
| GPU | `serve.py` (기본 bf16) 또는 HF 내보내기 후 vLLM |
| 평가 | `final_eval.py`·`sft_eval.py` (항상 fp32) |

```bash
# llama.cpp: OpenAI 호환 서버까지 한 번에
python export_gguf.py --ckpt checkpoints/sft_1b/latest.pt --out gguf/bebut-1.2b-sft.gguf --type q8_0
llama-server -m gguf/bebut-1.2b-sft.gguf -c 2048 --port 8080

# Ollama
cd gguf && ollama create bebut -f Modelfile && ollama run bebut

# 이 저장소의 서버 (웹 UI 포함)
python serve.py --ckpt checkpoints/sft_1b/latest.pt
```

## 시도했다가 접은 것: 추측 디코딩

같은 토크나이저로 학습한 125M 모델이 있어서, 그것을 초안(draft) 모델로 쓰는 추측
디코딩(speculative decoding)을 검토했습니다. 1.2B가 여러 토큰을 한 번에 검증하면 빨라지는
방식입니다. 실측해 보니 이 조합에선 이득이 거의 없었습니다.

| | 시간 (CPU, bf16) |
|---|---|
| 1.2B 1토큰 | 83ms |
| 1.2B 5토큰 한 번에 | 128ms |
| 125M 1토큰 (초안) | 18ms |

125M은 파라미터의 40%가 64K vocab 임베딩이라 1.2B 대비 22%나 비싸고, CPU에선 5토큰 검증이
1토큰의 1.5배입니다. 초안 4토큰 기준 기대 속도는 수용률 0.5·0.6·0.7·0.8에서 각각
0.8·1.0·1.2·1.4배라 구현 복잡도에 비해 얻는 게 작습니다. vocab이 작은 초안 모델을 따로
학습하지 않는 한 Q8_0 양자화가 훨씬 확실한 개선입니다.

## 다시 재기

```bash
python bench.py --config base_1b          # 학습 처리량 (GPU)
```

위 추론 표는 `configs/base_1b`로 무작위 모델을 만들어 PyTorch fp32·bf16 디코드, 프리픽스
캐시 TTFT, `export_gguf.py`로 내보낸 F16·Q8_0을 llama-cpp-python 0.3.36(`n_threads=4`)으로
잰 값입니다. Q8_0의 출력 차이는 정상 초기화 가중치에서 logit 표준편차의 5% 이내였습니다
(학습된 모델로는 직접 재지 못했습니다 — 보통 손실이 거의 없지만, 품질이 중요한 평가는 이
저장소의 fp32 경로로 하세요).
