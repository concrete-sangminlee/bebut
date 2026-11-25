# 추론 가이드 — 어디서 어떻게 돌릴까

1.2B 모델을 돌리는 방법은 세 가지입니다. 이 문서의 속도는 모두 **1.2B 실제 크기**
(`configs/base_1b`, 무작위 가중치 — 속도는 가중치 값과 무관)를 `bench_infer.py`로 잰 실측입니다
(원본 `logs/inference_bench.json`).

측정 환경: Intel Xeon 2.3GHz **4코어**(AVX512-BF16·AMX 지원), 메모리 15GB, GPU 없음, 다른 부하가
없는 상태. 값마다 3번 재 중앙값을 썼고, 전체를 두 번 돌렸을 때 값의 차이는 5% 이내였습니다. 코어가 더
많거나 메모리 대역폭이 큰 기계에선 전부 빨라지지만 순서는 같습니다.

## 디코드 속도 (토큰 생성)

| 경로 | 형식 | 크기 | 속도 |
|---|---|---|---|
| PyTorch (`sample.py`, `serve.py`) | fp32 | 4.8GB | 8.3 tok/s |
| PyTorch | bf16 | 2.4GB | 12.4 tok/s |
| llama.cpp (`export_gguf.py`) | F16 | 2.40GB | 14.3 tok/s |
| **llama.cpp** | **Q8_0** | **1.28GB** | **15.9 tok/s** |
| llama.cpp | Q4_0 | 0.74GB | 18.8 tok/s |

디코드는 토큰마다 가중치 전체를 메모리에서 읽어야 해서 **가중치가 작을수록 빠릅니다**. bf16(절반)이
fp32보다 1.5배, 8비트 Q8_0이 1.9배, 4비트 Q4_0이 2.2배 빠릅니다. 다만 크기에 비례하지는 않습니다 —
Q8_0은 F16의 절반 크기지만 11% 빠르고, Q4_0은 Q8_0보다 18% 빠릅니다(값을 푸는 계산이 늘어서).
같은 16비트끼리는 llama.cpp F16이 PyTorch bf16보다 15% 빠릅니다.

`sample.py`·`serve.py`는 CPU에 bf16 명령(AVX512-BF16) 또는 AMX가 있으면 **자동으로 bf16**을
씁니다. AVX-512만 있는 CPU(Skylake·Ice Lake 등)에선 bf16을 흉내 내느라 prefill이 fp32보다
몇 배 느려지므로 fp32를 씁니다 (`--dtype fp32|bf16`으로 강제할 수 있습니다). 평가 스크립트는 점수가 dtype에 흔들리지 않게 항상 fp32입니다.

## 양자화는 품질을 얼마나 깎나

속도만 보면 Q4_0이 가장 좋지만, 양자화는 출력을 바꿉니다. 학습된 가중치에는 양자화 오차를 키우는
큰 값이 있어서 무작위 가중치로는 이것을 볼 수 없습니다. 이 환경에는 학습된 bebut 체크포인트가
없으므로, **구조가 같은(Qwen3 계열) 학습된 공개 모델 Qwen3-1.7B-Base**를 이 저장소의 모델 코드로
읽어(`export_hf.import_state_dict`) 이 저장소의 양자화기로 GGUF를 만들고, llama.cpp로 같은 글의
perplexity를 쟀습니다 (`evals/quant_quality.py`, 원본 `logs/quant_quality.json`). 글은 FineWeb-2
한국어 test split 60KB, 256토큰 묶음 63개(채점 16,065토큰), 기준은 PyTorch fp32입니다.

| 형식 | 만든 곳 | 크기 | 가중치당 비트 | perplexity | 기준 대비 | 1순위 예측 일치 |
|---|---|---|---|---|---|---|
| fp32 (기준) | PyTorch | 6.9GB | 32 | 11.242 | — | — |
| F16 | 이 저장소 | 3.45GB | 16.0 | 11.242 | −0.00% | 100.0% |
| **Q8_0** | 이 저장소 | 1.83GB | 8.5 | 11.273 | **+0.28%** | 96.8% |
| Q6_K | llama.cpp | 1.42GB | 6.6 | 11.505 | +2.3% | 92.2% |
| Q5_K_M | llama.cpp | 1.26GB | 5.8 | 11.701 | +4.1% | 90.3% |
| Q4_K_M | llama.cpp | 1.11GB | 5.1 | 12.233 | +8.8% | 84.6% |
| Q4_0 | 이 저장소 | 1.13GB | 5.3 | 12.695 | **+12.9%** | 80.4% |

- **F16은 손실이 없고, Q8_0은 perplexity 0.3% 수준**입니다. 다만 Q8_0도 다음 토큰의 1순위 예측이
  30번에 한 번꼴로 달라지므로(96.8%), greedy로 생성한 글이 fp32와 글자까지 같지는 않습니다.
- **4비트는 이 크기의 모델에서 비쌉니다.** Q4_0은 perplexity가 13% 오르고 1순위 예측의 20%가
  바뀝니다. 흔히 인용되는 "4비트는 거의 무손실"은 7B 이상 모델에서 나온 말이고, 이 측정의 1.7B 모델은
  훨씬 민감했습니다.
- 같은 4비트대라면 llama.cpp의 K-양자화가 낫습니다(Q4_K_M +8.8% vs Q4_0 +12.9%, 크기는 비슷).
  이 저장소는 Q4_0까지만 직접 구현했으므로, 8비트보다 작은 파일이 필요하면 **F16으로 내보낸 뒤
  llama.cpp의 `llama-quantize`로 Q6_K나 Q5_K_M을 만드는 쪽**을 권합니다.
- 이 수치는 llama.cpp 추론 경로가 아니라 **양자화 자체의 손실**입니다. 같은 양자화 가중치를 PyTorch
  모델에 넣어 재도 Q8_0 +0.19%, Q4_0 +12.8%가 나옵니다(`--torch-check`). 남는 차이는 llama.cpp(ggml)가
  양자화 가중치와 곱할 때 활성값도 8비트 블록으로 바꾸는 데서 오는 것으로 보입니다.
- **bebut 자체의 수치는 아닙니다.** 구조와 크기가 비슷한 다른 모델의 결과이고, 가중치 분포가 다르면
  민감도도 다릅니다. 체크포인트가 있으면 직접 잽니다(`./remeasure.sh`의 4번 단계도 이것을 돌립니다).

```bash
python -m evals.quant_quality --ckpt checkpoints/sft_1b/latest.pt --out logs/quant_quality_bebut.json
python -m evals.quant_quality --llama-types q4_k_m q5_k_m q6_k --torch-check --out logs/quant_quality.json   # 위 표
```

## 대화가 길어질 때 — 프리픽스 캐시

대화는 턴마다 기록 전체를 다시 넣습니다. 기록 1,000토큰 + 새 질문 60토큰일 때 첫 토큰까지:

| | 첫 토큰까지 |
|---|---|
| 캐시 없이 (매 턴 1,060토큰 전부 계산) | 1.39초 |
| **프리픽스 캐시** (지난 턴과 달라진 60토큰만) | **0.23초** |

(PyTorch bf16. 대화가 문맥 창 2,048토큰을 다 채운 뒤에도, 오래된 턴을 창의 1/4 단위로 묶어 버리기
때문에 80턴 모의 대화에서 턴마다 평균 1,187토큰을 캐시에서 재사용하고 315토큰만 새로 계산합니다.)

`sample.py --chat`과 `serve.py`는 기본으로 켜져 있고, 서버 응답의
`usage.prompt_tokens_details.cached_tokens`와 웹 UI의 "문맥 N토큰 중 M토큰 캐시 재사용"으로
확인할 수 있습니다. fp32에선 결과가 캐시 없이 생성한 것과 토큰 단위로 같습니다. bf16에선
반올림 차이로 드물게 달라질 수 있어, 완전한 재현이 필요하면 `--no-prefix-cache`로 끕니다.

## 권장 구성

| 상황 | 권장 |
|---|---|
| CPU에서 대화·서빙 (학습 중이라 GPU를 비워둬야 할 때 포함) | **GGUF Q8_0 + llama.cpp / Ollama** — fp32의 1.9배 속도, perplexity 손실 0.3% 수준 |
| Q8_0(1.28GB)보다 작아야 할 때 | F16으로 내보낸 뒤 `llama-quantize`로 Q6_K · Q5_K_M (직접 뽑는 Q4_0은 가장 작고 빠르지만 손실이 가장 크다) |
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

## 내보낸 모델(HF · GGUF)을 쓸 때 주의

- **입력을 NFC로 정규화하세요.** 이 저장소의 대화 CLI·서버는 메시지를 NFC로 바꾸지만, 내보낸
  토크나이저는 원본과 바이트 단위로 같아야 해서 정규화하지 않습니다. 맥 파일 이름이나 PDF에서 온
  NFD 한글(자모 분리)은 학습 때 본 적 없는 바이트 조각 토큰이 됩니다(한 문장 9 → 151토큰).
- assistant 기록이 공백·줄바꿈으로 시작하는 경우에만, 내장 chat_template의 토큰 경계가 SFT와
  달라질 수 있습니다. 보통의 대화에서는 토큰까지 같습니다.
- 생성은 `<|im_end|>`와 `<|eos|>` 둘 다에서 멈춥니다(HF generation_config, GGUF의 eot·eom, Ollama
  Modelfile의 stop).

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
python bench_infer.py --config base_1b --out logs/inference_bench.json   # 이 문서의 속도 표 전부 (메모리 약 8GB, 4분)
python bench_infer.py --config mid_125m --types                          # 작은 모델로 빠르게, GGUF 생략
python bench.py --config base_1b                                         # 학습 처리량 (GPU)
```

`bench_infer.py`는 설정 크기의 무작위 모델로 PyTorch fp32·bf16 디코드, 프리픽스 캐시 유무의 첫 토큰
시간, 샘플러, 문맥 창 재사용, 그리고 `export_gguf.py`로 내보낸 F16·Q8_0·Q4_0의 llama.cpp 디코드
(llama-cpp-python, 없으면 생략)를 잽니다. 값마다 여러 번 재 중앙값을 쓰고, CPU·스레드 수·시작 시점의
부하를 결과에 남깁니다 — 다른 작업이 CPU를 쓰는 중에 재면 30~40% 낮게 나오기 때문입니다(시작 시점
부하가 1을 넘으면 경고합니다). 예전 표의 값은 손으로 잰 것이라 지금 값과 몇 %씩 다릅니다
(F16 12.6 → 14.3, Q8_0 16.4 → 15.9 tok/s, 첫 토큰 2.04 → 1.39초).
