#!/bin/bash
# 체크포인트가 있는 서버에서 한 번 돌리면, 아직 재지 못한 수치(docs/RESULTS.md §6)를 순서대로 만든다.
#
#   1. 사전학습 평가   채점 버그를 고친 뒤의 KOBEST·KMMLU·HAE-RAE, annealing 직전/최종 → logs/final_eval_v2.json
#   2. SFT 평가       베이스 · 에폭0 · 최종 (대화 문항 + KOBEST)                      → logs/sft_eval_v2.json
#   3. 원문 기준 BPB  공개 모델 기준값(logs/text_bpb.json)과 같은 글 · 한 표로         → logs/text_bpb_bebut.json
#   4. 양자화 품질    F16 · Q8_0 · Q4_0이 perplexity를 얼마나 바꾸나 (llama-cpp-python 필요) → logs/quant_quality_bebut.json
#   끝으로 annealing 직전/최종, 베이스/SFT를 같은 문항끼리 짝 비교(McNemar)해 차이가 표본 오차보다 큰지 출력한다.
#
# 결과 파일이 이미 있는 단계는 건너뛴다 — 중간에 끊겨도 다시 부르면 이어서 한다. 한 단계가
# 실패하거나 체크포인트가 없어도 나머지는 계속하고, 끝에 못 한 단계를 모아 알려준다.
#
# 사용: ./remeasure.sh                  # GPU가 비어 있을 때
#       DEVICE=cpu ./remeasure.sh       # 학습이 GPU를 쓰는 중이면 (훨씬 느리다 — 1번에 LIMIT=500 권장)
#       DRY_RUN=1 ./remeasure.sh        # 돌릴 명령만 출력
# 환경변수: PY(파이썬), DEVICE(auto), OUT(logs), LIMIT(과제별 문항 수 제한 — 기본은 전체),
#          PRE · BASE · SFT_E0 · SFT(체크포인트 경로)
set -u
cd "$(dirname "$0")"   # 저장소 위치에 의존하지 않는다

PY="${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}"
DEVICE="${DEVICE:-auto}"
OUT="${OUT:-logs}"
BASE="${BASE:-checkpoints/base_1b/latest.pt}"
SFT_E0="${SFT_E0:-checkpoints/sft_1b/epoch0.pt}"
SFT="${SFT:-checkpoints/sft_1b/latest.pt}"
if [ -z "${PRE:-}" ]; then   # annealing 직전 스냅샷: 지금의 train.py는 19,199, 원래 실행은 19,200 (final_eval.py와 같은 순서)
  PRE=checkpoints/base_1b/step_019199.pt
  [ -f "$PRE" ] || PRE=checkpoints/base_1b/step_019200.pt
fi
LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")
BASELINE_ARGS=()   # 공개 모델 기준값이 있으면 같은 표에 놓는다
[ -f logs/text_bpb.json ] && BASELINE_ARGS=(--baseline logs/text_bpb.json)
PPL_ARGS=()   # val shard가 없는 기계에서는 perplexity만 빼고 잰다
if [ ! -d data/shards_26b ] || [ ! -d data/shards_anneal ]; then
  echo "data/shards_26b · data/shards_anneal이 없다 — 1번에서 perplexity·BPB(val shard)는 건너뛴다"
  PPL_ARGS=(--skip-ppl)
fi

SKIPPED=()
FAILED=()

show() {  # 명령을 읽기 좋게 출력한다 (공백이 든 인자만 따옴표로)
  local a
  printf ' '
  for a in "$@"; do
    case "$a" in
      *" "*) printf ' "%s"' "$a" ;;
      *) printf ' %s' "$a" ;;
    esac
  done
  echo
}

# step <이름> <결과 파일> <필요한 체크포인트...> -- <명령...>
step() {
  local name="$1" out="$2" missing=()
  shift 2
  while [ "$1" != "--" ]; do
    [ -f "$1" ] || missing+=("$1")
    shift
  done
  shift
  if [ -s "$out" ]; then
    echo "[건너뜀] $name — $out 이 이미 있다"
    return 0
  fi
  echo
  echo "[$(date +%F\ %T)] $name"
  show "$@"
  [ -n "${DRY_RUN:-}" ] && return 0
  if [ ${#missing[@]} -gt 0 ]; then
    echo "  체크포인트가 없어 건너뛴다: ${missing[*]}"
    SKIPPED+=("$name")
    return 0
  fi
  "$@" || FAILED+=("$name")
}

step "1. 사전학습 평가" "$OUT/final_eval_v2.json" "$PRE" "$BASE" -- \
  "$PY" final_eval.py --ckpts "annealing 직전=$PRE" "최종=$BASE" --exclude-ids evals/contaminated_ids.json \
  --device "$DEVICE" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} ${PPL_ARGS[@]+"${PPL_ARGS[@]}"} \
  --out "$OUT/final_eval_v2.json"

step "2. SFT 평가" "$OUT/sft_eval_v2.json" "$BASE" "$SFT_E0" "$SFT" -- \
  "$PY" sft_eval.py --models "베이스=$BASE" "SFT 에폭0=$SFT_E0" "SFT 최종=$SFT" \
  --exclude-ids evals/contaminated_ids.json --device "$DEVICE" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
  --out "$OUT/sft_eval_v2.json" --answers-out "$OUT/sft_eval_answers_v2.md"

step "3. 원문 기준 BPB" "$OUT/text_bpb_bebut.json" "$PRE" "$BASE" "$SFT" -- \
  "$PY" -m evals.text_bpb --ckpt "annealing 직전=$PRE" "최종=$BASE" "SFT 최종=$SFT" --kb 300 \
  --device "$DEVICE" ${BASELINE_ARGS[@]+"${BASELINE_ARGS[@]}"} --out "$OUT/text_bpb_bebut.json"

if [ -n "${DRY_RUN:-}" ] || "$PY" -c "import llama_cpp" 2> /dev/null; then
  step "4. 양자화 품질" "$OUT/quant_quality_bebut.json" "$SFT" -- \
    "$PY" -m evals.quant_quality --ckpt "$SFT" --out "$OUT/quant_quality_bebut.json"
else
  echo
  echo "[건너뜀] 4. 양자화 품질 — llama-cpp-python이 없다 (설치 방법: requirements-dev.txt)"
  SKIPPED+=("4. 양자화 품질")
fi

# 차이가 표본 오차보다 큰가 — 같은 문항끼리 짝 비교
compare() {  # compare <결과 파일> <기준 라벨> <비교 라벨>
  [ -s "$1" ] || [ -n "${DRY_RUN:-}" ] || return 0
  echo
  show "$PY" compare_evals.py "$1" --a "$2" --b "$3"
  [ -n "${DRY_RUN:-}" ] && return 0
  "$PY" compare_evals.py "$1" --a "$2" --b "$3" || FAILED+=("비교: $1")
}
compare "$OUT/final_eval_v2.json" "annealing 직전" "최종"
compare "$OUT/sft_eval_v2.json" "베이스" "SFT 최종"

echo
[ ${#SKIPPED[@]} -gt 0 ] && printf '건너뛴 단계: %s\n' "${SKIPPED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '실패한 단계: %s\n' "${FAILED[@]}"
  exit 1
fi
echo "끝. 결과는 $OUT/ 아래에 있다 — 수치를 docs/RESULTS.md에 옮길 때 † 표시(예전 채점기) 값과 구분해 적는다."
