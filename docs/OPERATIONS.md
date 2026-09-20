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

**저장은 원자적으로** 합니다. 14GB 파일을 직접 덮어쓰면 저장 중 전원이 끊겼을 때
체크포인트가 깨져 학습 전체를 잃습니다. 임시 파일에 쓴 뒤 `os.replace`로 이름만 바꿉니다.

실제로 이 설계 덕분에 2주간 중단했다가 재개했을 때 loss가 중단 시점과 같은 수준에서
그대로 이어졌습니다.

## 손실 없이 멈추기

GPU를 양보해야 할 때는 **체크포인트 저장 직후**를 노려 멈추면 진행분을 잃지 않습니다.

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
