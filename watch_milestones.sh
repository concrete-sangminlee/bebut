#!/bin/bash
# 새 스냅샷(step_*.pt)이 생기면 평가한다. eval_milestones.py가 이미 평가한 step은 건너뛰고
# (읽지 못하는 스냅샷도 건너뛰고 다음에 다시 본다) 할 일이 없으면 바로 끝나므로 매시간 그냥
# 부른다 — 예전엔 스냅샷 수와 milestones.csv 행 수를 비교해서, 중복 행이 한 번 생기면 새
# 스냅샷을 영영 감지하지 못했다. CPU 전용이라 GPU 학습과 무관.
#
# 사용: ./watch_milestones.sh [config]
cd "$(dirname "$0")"   # 저장소 위치에 의존하지 않는다
CONFIG="${1:-base_1b}"
while true; do
  if ls "checkpoints/$CONFIG"/step_*.pt > /dev/null 2>&1; then
    echo "[$(date +%F\ %T)] 스냅샷 확인"
    EVAL_THREADS=16 .venv/bin/python eval_milestones.py --config "$CONFIG" --limit 150 --shots 2 \
      >> milestones.log 2>&1
  fi
  sleep 3600
done
