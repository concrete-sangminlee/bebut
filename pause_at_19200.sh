#!/bin/bash
# [기록용 — 지금은 필요 없다] 1.2B 학습 때 step 19,200(= annealing 전환 지점)에서 학습을
# 멈추고 latest.pt를 step_019200.pt로 복사한 스크립트. RESULTS.md §2의 '19,200' 기준점이
# 이렇게 만들어졌다. 엄밀히는 step 19,200 체크포인트는 annealing 데이터로 한 step을 학습한
# 뒤의 상태다(전환은 그 step을 학습하기 전에 일어난다).
#
# 지금의 train.py는 전환 직전 step(anneal_from_step - 1 = 19,199)을 자동으로 스냅샷
# (step_019199.pt)한다. 새로 학습한다면 이 스크립트 대신 그 스냅샷을 쓰면 된다 — 이걸 같이
# 돌리면 스냅샷이 하나 더 생기고 마일스톤 평가도 중복된다.
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
