#!/bin/bash
# 새 스냅샷(step_*.pt)이 생기면 평가한다. eval_milestones.py가 이미 평가한 step은
# 건너뛰므로 주기적으로 돌려도 중복 작업이 없다. CPU 전용이라 GPU 학습과 무관.
cd "$(dirname "$0")"   # 저장소 위치에 의존하지 않는다
while true; do
  n=$(ls checkpoints/base_1b/step_*.pt 2>/dev/null | wc -l)
  done_n=$(( $(wc -l < checkpoints/base_1b/milestones.csv 2>/dev/null || echo 1) - 1 ))
  if [ "$n" -gt "$done_n" ]; then
    echo "[$(date +%F\ %T)] 새 스냅샷 감지 ($done_n → $n), 평가 시작"
    EVAL_THREADS=16 .venv/bin/python eval_milestones.py --config base_1b --limit 150 --shots 2 \
      >> milestones.log 2>&1
    echo "[$(date +%F\ %T)] 평가 완료"
  fi
  sleep 3600
done
