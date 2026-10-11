#!/bin/bash
# run_forgetting.sh — eval loss 망각 행렬(11시점 × 5 검증셋) 런 단위 디스패처.
#   CUDA_VISIBLE_DEVICES=0 bash run_forgetting.sh llava_order_14203 qwen25vl_7b qwen25vl_7b_reverse
#   bash run_forgetting.sh --dry vila_3b_order_20431
# run 키: forward=<model>, reverse=<model>_reverse, 무작위=<model>_order_<tag>  (model: llava|mipha|vila_3b|vila_8b|openflamingo|qwen25vl_7b)
# 결과: results_<model>/<run_key>__<tag>.json  (fwd/rev 는 기존대로 fwd__<tag> / rev__<tag>). 이미 있는 상태는 건너뜀.
# 검증셋: S1 은 무작위 런에서 nohex (fwd/rev 기존 결과는 hex 시절). 로그: logs_forgetting/<run_key>.log
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"; mkdir -p logs_forgetting
export PYTHONUNBUFFERED=1
DRY=0; RUNS=()
for a in "$@"; do case "$a" in --dry) DRY=1 ;; -h|--help) sed -n '2,8p' "$0"; exit 0 ;; -*) echo "[ERR] 옵션: $a" >&2; exit 1 ;; *) RUNS+=("$a") ;; esac; done
[ "${#RUNS[@]}" -ge 1 ] || { echo "usage: CUDA_VISIBLE_DEVICES=<id> $0 [--dry] <run_key> [...]" >&2; exit 1; }
if (( DRY == 0 )); then
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "[ERR] CUDA_VISIBLE_DEVICES 를 지정하세요" >&2; exit 1; }
  [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-2]$ ]] || { echo "[ERR] GPU 하나만 (0|1|2): ${CUDA_VISIBLE_DEVICES}" >&2; exit 1; }
fi
for v in $(env | grep -oE "^(CONDA_[A-Z_]+|_CE_[A-Z]+)=" | tr -d =); do unset "$v"; done
source "$HOME/anaconda3/etc/profile.d/conda.sh"

resolve() {   # run 키 → ENV, RUNNER, RUNNER_ARGS, MODE(fwd|rev|order)
  local run="$1"
  case "$run" in
    llava*)        ENV=llava;        RUNNER=run_forgetting_llava.sh;        RUNNER_ARGS=() ;;
    mipha*)        ENV=mipha;        RUNNER=run_forgetting_mipha.sh;        RUNNER_ARGS=() ;;
    vila_3b*)      ENV=vila;         RUNNER=run_forgetting_vila.sh;         RUNNER_ARGS=(3b) ;;
    vila_8b*)      ENV=vila;         RUNNER=run_forgetting_vila.sh;         RUNNER_ARGS=(8b) ;;
    openflamingo*) ENV=openflamingo; RUNNER=run_forgetting_openflamingo.sh; RUNNER_ARGS=() ;;
    qwen25vl_7b*)  ENV=qwenvl;       RUNNER=run_forgetting_qwen25vl.sh;     RUNNER_ARGS=() ;;
    *) echo "[ERR] 모델 판별 불가: $run" >&2; return 1 ;;
  esac
  if [[ "$run" == *_order_* ]]; then MODE=order; elif [[ "$run" == *_reverse ]]; then MODE=rev; else MODE=fwd; fi
}

FAILED=()
for RUN in "${RUNS[@]}"; do
  resolve "$RUN" || { FAILED+=("$RUN"); continue; }
  LOG="logs_forgetting/${RUN}.log"
  ENVV=(MASTER_PORT=${MASTER_PORT:-$((25100 + ${CUDA_VISIBLE_DEVICES:-0}*10))})   # 큐마다 다른 포트 (같은 GPU 에 큐 2개면 MASTER_PORT 지정)
  if [[ "$MODE" == "order" ]]; then
    ROOT=$(python3 /home/hail/vlm3/order_runs.py root "$RUN") || { echo "[ERR] $RUN 체크포인트 없음"; FAILED+=("$RUN"); continue; }
    ENVV+=(RUN_KEY="$RUN" ORDER_ROOT="$ROOT")
  else
    ENVV+=(DIRS="$MODE")
  fi
  echo "============================================================"
  echo "[FORGET] run=${RUN} env=${ENV} gpu=${CUDA_VISIBLE_DEVICES:-none} mode=${MODE} → env ${ENVV[*]} bash ${RUNNER} ${RUNNER_ARGS[*]}"
  echo "============================================================"
  if (( DRY )); then
    if [[ "$MODE" == "order" ]]; then
      if [[ "$ENV" == openflamingo ]]; then python3 /home/hail/vlm3/order_runs.py states "$ROOT" --base BASE --mid checkpoint-1562 --end . | sed 's/^/   /'
      else python3 /home/hail/vlm3/order_runs.py states "$ROOT" --base BASE | sed 's/^/   /'; fi
    fi
    continue
  fi
  conda activate "$ENV" 2>/dev/null || { echo "[ERR] env $ENV 실패"; FAILED+=("$RUN"); continue; }
  echo "[FORGET] $(date '+%F %T') START run=${RUN}" >> "$LOG"
  env "${ENVV[@]}" bash "$RUNNER" "${RUNNER_ARGS[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  # 결과 파일 수 확인 (11 이어야)
  case "$RUN" in vila_3b*) RD=results_vila_3b ;; vila_8b*) RD=results_vila_8b ;; qwen25vl_7b*) RD=results_qwen25vl_7b ;; *) RD="results_${RUN%%_*}" ;; esac
  prefix=$([[ "$MODE" == order ]] && echo "$RUN" || echo "$MODE")
  n=$(ls "$RD"/${prefix}__*.json 2>/dev/null | wc -l)
  echo "[FORGET] $(date '+%F %T') END run=${RUN} rc=${rc} results=${n}/11" | tee -a "$LOG"
  (( n == 11 )) || FAILED+=("$RUN")
  conda deactivate 2>/dev/null || true
done
echo "============================================================"
if (( ${#FAILED[@]} )); then echo "[FORGET] 미완/실패: ${FAILED[*]}"; exit 1; fi
echo "[FORGET] 완료: ${RUNS[*]}  ($(date '+%m-%d %H:%M'))  → python make_eval_loss_table.py"
