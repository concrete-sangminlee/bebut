#!/bin/bash
# step 19,200(= annealing 전환 지점)에서 학습을 멈추고 그 상태를 스냅샷으로 남긴다.
#
# 19,200은 마일스톤 주기(4,000)에 걸리지 않아 자동 스냅샷이 없다. 대신 100 step마다
# 저장되는 latest.pt를 그 시점에 복사해 step_019200.pt로 보존한다.
# 이 체크포인트는 고품질 데이터를 보기 직전 상태라 annealing 효과의 비교 기준이 된다.
cd "$(dirname "$0")"   # 저장소 위치에 의존하지 않는다
echo "[$(date +%F\ %T)] step 19,200 대기 시작"

# 학습 로그에 해당 step의 체크포인트 저장이 찍히면 latest.pt가 그 시점이다
until grep -q "체크포인트 저장: checkpoints/base_1b/latest.pt (step 19200)" train_base_1b.log; do
  sleep 120
done
sleep 30   # 14GB 쓰기 완료 여유 (원자적 저장이라 rename 이후엔 안전)

echo "[$(date +%F\ %T)] step 19,200 도달 — 학습 중단"
pgrep -f "[r]un_training.sh" | xargs -r kill      # 감시 먼저 (안 그러면 자동 재시작)
pgrep -f "[w]atch_milestones" | xargs -r kill
sleep 3
pgrep -f "[t]rain.py --config base_1b" | xargs -r kill
sleep 10
pgrep -f "[t]rain.py" | xargs -r kill -9 2>/dev/null
sleep 5
echo "[$(date +%F\ %T)] GPU 해제: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

cp checkpoints/base_1b/latest.pt checkpoints/base_1b/step_019200.pt
echo "[$(date +%F\ %T)] 스냅샷 보존: step_019200.pt (annealing 직전 상태)"

echo "[$(date +%F\ %T)] 마일스톤 평가 시작 (CPU)"
EVAL_THREADS=32 PYTHONUNBUFFERED=1 .venv/bin/python eval_milestones.py \
    --config base_1b --limit 150 --shots 2 >> milestones.log 2>&1
echo "[$(date +%F\ %T)] 평가 완료 — 일시정지 상태"
cat checkpoints/base_1b/milestones.csv
