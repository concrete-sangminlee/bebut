# 운영 가이드 — 장기 학습을 무인으로 버티기

1.2B 사전학습은 GPU 한 장으로 약 6주가 걸렸고, 그 사이 정전·GPU 경합·네트워크 단절을
겪었습니다. 이 문서는 그걸 버티기 위해 만든 장치들과 실제 운영 절차입니다.

## 실행과 재개

```bash
./run_training.sh base_1b data/shards_26b    # 사전학습 (감시 포함)
./watch_milestones.sh &                       # 스냅샷 생길 때마다 자동 평가 (CPU)
./resume.sh                                   # 중단된 학습 재개 (latest.pt에서)
```

`run_training.sh`는 학습이 비정상 종료하면 **마지막 체크포인트에서 자동 재개**합니다
(최대 200회). 10분 안에 죽는 일이 3번 연속되면 30분 대기하며 `nvidia-smi` 결과를
로그에 남깁니다 — GPU가 점유된 상태에서 무의미한 재시도를 반복하지 않기 위해서입니다.

## 체크포인트 설계

학습 재개가 매끄러우려면 세 가지가 모두 저장돼야 합니다.

```
model        가중치
optimizer    AdamW momentum·variance   ← 없으면 재개 시 loss가 튄다
train_loader (seed, step)              ← 데이터 순서 재현
```

데이터로더의 무작위성을 `(seed, step)` 두 숫자로만 결정되게 만들어서, step 번호만
저장하면 중단 지점의 데이터 순서가 정확히 재현됩니다.

**annealing 구간의 재개**는 따로 주의가 필요했습니다. 체크포인트의 로더 상태는 저장 당시
쓰던 로더(annealing 이후면 고품질 데이터 로더)의 것인데, 예전 코드는 재개 직후 annealing
로더를 새로 만들면서 위치를 step 번호로 덮어써 데이터 스트림을 되감았습니다. 지금은 마지막
step이 `anneal_from_step` 이상이면 annealing 로더를 만들고 저장된 위치를 그대로 복원합니다.
`tests/test_train_e2e.py`가 "중단 없이 학습"과 "annealing 앞뒤에서 두 번 끊고 재개"의 최종
가중치가 비트 단위로 같은지 검사합니다.

**데이터 샘플링 방식**은 새 학습부터 `epoch`(겹치지 않는 윈도우를 에폭마다 섞어 모든 토큰을
한 번씩)이 기본입니다. 예전 체크포인트는 상태에 `mode`가 없어서 자동으로 `random`(복원추출)
그대로 재개됩니다. 설정에서 `sampling="random"`으로 예전 방식을 고를 수도 있습니다.

**비정상 step**: loss나 gradient norm이 NaN/inf면 그 step의 업데이트만 건너뛰고 경고를
남깁니다. 5번 연속이면 프로세스를 끝내 감시 스크립트가 마지막 체크포인트에서 다시 띄웁니다.
`metrics.csv`에는 step마다 `grad_norm`, 10 step마다 `tokens_per_sec`이 기록됩니다
(예전 형식의 파일에 이어 쓸 때는 예전 열만 씁니다).

**저장은 원자적으로** 합니다. 14GB 파일을 직접 덮어쓰면 저장 중 전원이 끊겼을 때
체크포인트가 깨져 학습 전체를 잃습니다. 임시 파일에 쓴 뒤 `os.replace`로 이름만 바꿉니다.

실제로 이 설계 덕분에 2주간 중단했다가 재개했을 때 loss가 중단 시점과 같은 수준에서
그대로 이어졌습니다.

## 손실 없이 멈추기

가장 간단한 방법은 `--stop-after`입니다. 그 step을 마치면 체크포인트를 저장하고 정상
종료(코드 0)하므로 `run_training.sh`도 재시작하지 않습니다.

```bash
python train.py --config base_1b --shard-dir data/shards_26b \
    --resume checkpoints/base_1b/latest.pt --stop-after 19199
```

annealing 전환 직전 step(`anneal_from_step - 1`)은 `milestone_every`가 설정돼 있으면
자동으로 `step_XXXXXX.pt` 스냅샷이 남습니다. 예전에는 이걸 위해 아래처럼 학습을 멈추고
손으로 복사했습니다.

이미 돌고 있는 학습을 멈춰야 할 때는 **체크포인트 저장 직후**를 노려 멈추면 진행분을 잃지 않습니다.

```bash
# 로그에 해당 step 저장이 찍히면 그때 종료
until grep -q "체크포인트 저장.*(step 19200)" train_base_1b.log; do sleep 120; done
pkill -f run_training.sh    # 감시부터 (안 그러면 자동 재시작됨)
sleep 3; pkill -f train.py
```

`pause_at_19200.sh`가 이 패턴의 실제 예입니다. **감시 스크립트를 먼저 종료**하는 것이
중요합니다.

## GPU를 공유할 때

이 GPU는 다른 연구(vLLM 추론 서버 14GB, RAG 파이프라인 15.7GB)와 공유됩니다.
그 워크로드는 예고 없이 뜹니다.

- 학습 VRAM은 **여유를 14GB 이상 남기고** 잡습니다. 배치 4(34.7GB)는 여유가 12GB뿐이라
  다른 작업이 뜰 때마다 OOM 크래시 루프에 빠졌습니다.
- 재개 전 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`로 확인합니다.
- **남의 프로세스를 죽이지 마세요.** 좀비로 보이는 15GB 프로세스가 실제로는 다른
  연구의 실행 중인 작업이었던 적이 있습니다. 죽이기 전에 `ps -o cmd= -p <pid>`로 확인합니다.

## 학습 중 안전하게 관찰하기

학습이 GPU를 쓰는 동안 평가나 샘플 생성은 **CPU로** 돌립니다.

```bash
CUDA_VISIBLE_DEVICES="" python gen_samples.py                    # 샘플 생성
CUDA_VISIBLE_DEVICES="" python eval_milestones.py --config base_1b --limit 150 --shots 2
```

1.2B 모델 기준 CPU forward는 컨텍스트 1,024토큰에 약 2초입니다. KOBEST 전체 문항은
CPU로 몇 시간 걸리므로 학습 중에는 `--limit`으로 줄이고, 최종 평가만 GPU로 돌립니다.

## 모델 서빙

```bash
python serve.py --ckpt checkpoints/sft_1b/latest.pt --port 8000          # 웹 UI + OpenAI 호환 API
python serve.py --ckpt ... --device cpu                                   # 학습 중엔 CPU로 (느리지만 GPU 무간섭)
```

요청은 한 번에 하나씩 처리합니다(락). 요청 사이에는 KV 캐시를 재사용합니다 — 같은 대화를
이어가는 요청은 기록 전체를 다시 보내지만, 지난 요청과 같은 앞부분은 계산하지 않습니다.
다른 대화가 끼어들면 공통 접두부가 짧아져 그만큼 다시 계산할 뿐 결과는 같습니다. 스트리밍 중 클라이언트가 끊으면 생성을 멈추고 락을
놓습니다. 인증이 없으므로 기본값은 `127.0.0.1`이고, 외부에 열 때(`--host 0.0.0.0`)는
신뢰하는 네트워크에서만 쓰세요. CUDA에선 기본 bf16(가중치 2.4GB), CPU에선 fp32입니다.

## DPO 돌리기

```bash
python dpo.py --base checkpoints/sft_1b/latest.pt --config dpo_1b --data prefs.jsonl
```

- **첫 로그의 `시작 val: loss`가 0.6931(ln 2)인지 먼저 보세요.** 시작 시 정책 = 참조 모델이라
  수학적으로 정확히 ln 2여야 합니다. 다르면 데이터 인코딩이나 모델 로드가 틀린 것입니다.
- 보상 정확도가 빠르게 1.0에 붙고 마진이 계속 커지면 과적합 신호입니다. 1.2B 기준
  `max_lr=5e-7`, `β=0.1`에서 시작하고, 응답 길이가 눈에 띄게 늘면 β를 올리거나 학습률을 낮춥니다.
- 메모리: 정책 학습 약 19GB + 참조 모델(bf16) 2.4GB + 활성값. SFT와 비슷한 26~30GB입니다.
- 결과 확인은 `sft_eval.py --models DPO=checkpoints/dpo_1b/latest.pt ...`로 SFT와 같은 조건에서.

## 진행 상황 확인

```bash
grep -E "step +[0-9]+ \|" train_base_1b.log | tail -1   # 진행률
grep -c "비정상 종료" train_base_1b.log                  # 크래시 횟수
cat checkpoints/base_1b/milestones.csv                   # 벤치마크 이력
```

**감시는 성공 신호와 실패 신호를 둘 다 잡아야 합니다.** "step 20에 도달하면 알림"으로만
걸었다가, 크래시 루프에 빠져 step 20에 영영 도달하지 못하는 동안 아무 알림도 오지 않은
적이 있습니다. 조용한 것이 정상이라는 보장은 없습니다.

## 실패 사례 기록

| 증상 | 원인 | 대응 |
|---|---|---|
| step 0에서 5회 연속 크래시 | 다른 작업이 GPU 15.7GB 점유 | 배치 4 → 2로 축소 |
| 로그가 몇 시간째 조용 | 출력 버퍼링 (리다이렉트 시) | `PYTHONUNBUFFERED=1`, 진행률·ETA 출력 추가 |
| 데이터 준비 ETA 43시간 | MinHash가 메인 프로세스 단일 스레드 | 해싱·필터를 워커로 이동 → 1.6시간 |
| 위키 3회 반복이 1회분만 남음 | 중복 제거가 의도적 반복을 무효화 | 소스별로 dedup on/off |
| 최종 스냅샷 누락 | 마지막 step(23,999)이 4,000의 배수가 아님 | 마지막 step도 스냅샷 저장 |
| annealing 재개 시 이미 본 배치 재학습 | 재개 직후 로더 위치를 step 번호로 덮어씀 | 저장된 로더 상태 복원 + e2e 비트 일치 테스트 |
| val loss가 ±0.1 출렁임 | 평가마다 다른 val 배치 | 매번 같은 val 배치 |
| 같은 설정인데 실행마다 다른 모델 | 초기화 시드 없음 | `seed` 설정 |
| KOBEST BoolQ가 한쪽으로만 답함 | 채점기가 선택지 첫 토큰을 빠뜨림 | 바이트 오프셋으로 경계 계산 (RESULTS.md §0) |
