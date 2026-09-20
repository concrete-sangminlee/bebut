#!/bin/bash
# 중단된 사전학습을 재개한다. checkpoints/<config>/latest.pt에서 이어지며,
# annealing 시작 지점을 지난 상태라면 재개 즉시 고품질 데이터로 전환된다.
#
# 사용: ./resume.sh [config] [shard-dir]
set -u
cd "$(dirname "$0")"   # 저장소 위치에 의존하지 않는다

CONFIG="${1:-base_1b}"
SHARD_DIR="${2:-data/shards_26b}"
CKPT="checkpoints/$CONFIG/latest.pt"

if pgrep -f "[t]rain.py --config $CONFIG" > /dev/null; then
  echo "이미 학습이 돌고 있습니다. 중복 실행하지 않습니다."
  exit 1
fi
if [ ! -f "$CKPT" ]; then
  echo "체크포인트가 없습니다: $CKPT"
  exit 1
fi

read -r step max <<EOPY
$(.venv/bin/python - "$CONFIG" "$CKPT" <<'PY'
import importlib, sys, torch
config, path = sys.argv[1], sys.argv[2]
step = torch.load(path, map_location="cpu", weights_only=False)["step"]
print(step, importlib.import_module(f"configs.{config}").train["max_steps"])
PY
)
EOPY

if [ -z "${step:-}" ]; then
  echo "체크포인트에서 step을 읽지 못했습니다: $CKPT"
  exit 1
fi
if [ "$step" -ge "$((max - 1))" ]; then
  # 실수로 완료된 학습을 다시 띄우면 로그만 지저분해진다
  echo "이미 완료된 학습입니다 (step $step / max_steps $max). 재개할 것이 없습니다."
  exit 0
fi

echo "체크포인트 step $step 에서 재개합니다 (다음 step: $((step + 1)), 목표 $max)"
nohup ./run_training.sh "$CONFIG" "$SHARD_DIR" > supervisor.log 2>&1 &
sleep 5
nohup ./watch_milestones.sh > watch_milestones.log 2>&1 &
sleep 3
echo "학습 감시 $(pgrep -fc '[r]un_training.sh')개 / 마일스톤 감시 $(pgrep -fc '[w]atch_milestones')개 기동"
echo "첫 진행 로그는 torch.compile 워밍업 때문에 20분쯤 뒤에 나옵니다."
