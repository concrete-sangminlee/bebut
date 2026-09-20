#!/bin/bash
# 일시정지한 학습을 재개한다. latest.pt에서 자동으로 이어지며,
# step 19,200 이후이므로 재개 즉시 annealing 고품질 데이터로 전환된다.
cd "$(dirname "$0")"   # 저장소 위치에 의존하지 않는다

if pgrep -f "[t]rain.py --config base_1b" > /dev/null; then
  echo "이미 학습이 돌고 있습니다. 중복 실행하지 않습니다."
  exit 1
fi

step=$(.venv/bin/python -c "
import torch
print(torch.load(checkpoints/base_1b/latest.pt, map_location=cpu, weights_only=False)[step])
" 2>/dev/null)
echo "체크포인트 step $step 에서 재개합니다 (다음 step: $((step+1)))"

nohup ./run_training.sh base_1b data/shards_26b > supervisor.log 2>&1 &
sleep 5
nohup ./watch_milestones.sh > watch_milestones.log 2>&1 &
sleep 3
echo "학습 감시 $(pgrep -fc "[r]un_training.sh")개 / 마일스톤 감시 $(pgrep -fc "[w]atch_milestones")개 기동"
echo "첫 진행 로그는 torch.compile 워밍업 때문에 20분쯤 뒤에 나옵니다."
