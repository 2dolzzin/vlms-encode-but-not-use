#!/bin/bash
# run_eval.sh — 커리큘럼 런 하나를 끝까지 평가: 5-task 추론 → blind 벤치 추론 → 채점 2종.
#
#   CUDA_VISIBLE_DEVICES=1 bash run_eval.sh llava_order_14203
#   CUDA_VISIBLE_DEVICES=1 bash run_eval.sh vila_3b_order_20431 mipha_order_31024   # 여러 런 순차
#   CUDA_VISIBLE_DEVICES=1 bash run_eval.sh llava_reverse                           # forward/reverse 도 동일
#   bash run_eval.sh --score-only llava_order_14203      # 추론 건너뛰고 채점만 (GPU 불필요)
#   bash run_eval.sh --dry llava_order_14203             # 실행할 명령만 출력
#   옵션: --no-5task / --no-blind  (한쪽만 돌릴 때)
#
# run 키 → (conda env, 스크립트) 는 접두사로 자동 판별. run 키는 평가 스크립트의 --model/--run 키와 같다:
#   llava, llava_reverse, llava_order_<tag> | mipha… | vila_3b…, vila_8b… | openflamingo… | qwen25vl_7b…
# 결과:
#   extract_5_tasks/result/<run>/inference/{raw_responses,accuracy,parsing}, extract_5_tasks/result/<run>/embedding
#   extract_benchmark/result/<run>_blind/{raw_responses,embeddings,accuracy,parsing}
#   로그: eval_logs/<run>.log
# 추론 스크립트는 (ckpt, task) 결과 파일이 있으면 건너뛰므로, 중간에 죽으면 같은 명령을 다시 치면 이어서 돈다.
# -u 는 쓰지 않는다: conda 의 activate.d 훅(qt-main_activate.sh)이 미정의 변수를 참조해서 set -u 아래서는
# 두 번째 런의 conda activate 에서 스크립트가 통째로 죽는다 (10/5 실제 발생). 변수는 아래서 직접 기본값 처리.
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1   # tee 로 파이프되면 print 가 블록 버퍼링돼 비정상 종료 시 마지막 출력이 유실됨 → 즉시 flush
LOG_DIR="${ROOT}/eval_logs"; mkdir -p "${LOG_DIR}"

DO_5TASK=1; DO_BLIND=1; SCORE_ONLY=0; DRY=0; RUNS=()
for a in "$@"; do
  case "$a" in
    --no-5task)   DO_5TASK=0 ;;
    --no-blind)   DO_BLIND=0 ;;
    --score-only) SCORE_ONLY=1 ;;
    --dry)        DRY=1 ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    -*)           echo "[ERR] 알 수 없는 옵션: $a" >&2; exit 1 ;;
    *)            RUNS+=("$a") ;;
  esac
done
[ "${#RUNS[@]}" -ge 1 ] || { echo "usage: CUDA_VISIBLE_DEVICES=<id> $0 [--no-5task] [--no-blind] [--score-only] [--dry] <run_key> [...]" >&2; exit 1; }

# GPU 는 반드시 명시 (지정 안 하면 GPU 0 을 잡아 학습과 충돌한 전례가 있음)
if (( SCORE_ONLY == 0 && DRY == 0 )); then
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "[ERR] CUDA_VISIBLE_DEVICES 를 지정하세요 (예: CUDA_VISIBLE_DEVICES=1)" >&2; exit 1; }
  [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-2](,[0-2])*$ ]] || { echo "[ERR] 이 장비의 GPU 는 0,1,2 입니다: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2; exit 1; }
fi

# run 키 → env / 스크립트 / 추론 인자
resolve() {
  local run="$1"
  case "$run" in
    llava*)        ENV=llava;        INF=inf_llava.py;        BEN=ben_llava_blind.py ;;
    mipha*)        ENV=mipha;        INF=inf_mipha.py;        BEN=ben_mipha_blind.py ;;
    vila_3b*)      ENV=vila;         INF=inf_vila.py;         BEN=ben_vila_3b_blind.py ;;
    vila_8b*)      ENV=vila;         INF=inf_vila.py;         BEN=ben_vila_8b_blind.py ;;
    openflamingo*) ENV=openflamingo; INF=inf_openflamingo.py; BEN=ben_openflamingo_blind.py ;;
    qwen25vl_7b*)  ENV=qwenvl;       INF=inf_qwen25vl.py;     BEN=ben_qwen25vl_blind.py ;;
    *) echo "[ERR] 모델을 판별할 수 없는 run 키: $run" >&2; return 1 ;;
  esac
  # 5-task 스크립트의 reverse 표기: llava/mipha/qwen 은 "<model>_reverse" 키, vila/openflamingo 는 --reverse 플래그
  INF_ARGS=(--model "$run")
  if [[ "$run" == *_reverse && ( "$INF" == inf_vila.py || "$INF" == inf_openflamingo.py ) ]]; then
    INF_ARGS=(--model "${run%_reverse}" --reverse)
  fi
}

run_step() {   # run_step <라벨> <작업디렉토리> <명령...>
  local label="$1" dir="$2"; shift 2
  echo; echo "──── [$label] $(date '+%m-%d %H:%M') ────"; echo "  (cd $dir && $*)"
  (( DRY )) && return 0
  ( cd "$dir" && "$@" ) 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  (( rc == 0 )) || { echo "[FAIL] $label (exit $rc)" | tee -a "$LOG"; return $rc; }
}

# 호출한 셸의 conda 상태(base 활성, 다른 env 활성, 반쯤 깨진 변수)와 무관하게 동작하도록 conda 변수를 비우고 새로 초기화.
# (CONDA_SHLVL 만 남고 CONDA_PREFIX 가 없는 식의 불일치 상태면 conda activate 자체가 죽는다)
for v in $(env | grep -oE "^(CONDA_[A-Z_]+|_CE_[A-Z]+)=" | tr -d =); do unset "$v"; done
source "$HOME/anaconda3/etc/profile.d/conda.sh"
FAILED=()
for RUN in "${RUNS[@]}"; do
  resolve "$RUN" || { FAILED+=("$RUN"); continue; }
  LOG="${LOG_DIR}/${RUN}.log"
  echo "============================================================"
  echo "[EVAL] run=${RUN}  env=${ENV}  gpu=${CUDA_VISIBLE_DEVICES:-none}  5task=${DO_5TASK} blind=${DO_BLIND} score_only=${SCORE_ONLY}"
  echo "  log → ${LOG}"
  echo "============================================================"
  (( DRY )) || echo "[EVAL] $(date '+%F %T') START run=${RUN} env=${ENV} gpu=${CUDA_VISIBLE_DEVICES:-none} 5task=${DO_5TASK} blind=${DO_BLIND} score_only=${SCORE_ONLY}" >> "$LOG"
  if (( !DRY )); then
    conda activate "$ENV" 2>/dev/null || { echo "[ERR] conda env '$ENV' 활성화 실패" >&2; FAILED+=("$RUN"); continue; }
    [ "$(basename "${CONDA_PREFIX:-}")" = "$ENV" ] || { echo "[ERR] env 전환 실패: CONDA_PREFIX=${CONDA_PREFIX:-}" >&2; FAILED+=("$RUN"); continue; }
    echo "  python = $(command -v python)"
  fi
  ok=1
  if (( SCORE_ONLY == 0 )); then
    # 체크포인트 인식 확인 (11시점이 아니면 중단 — 학습 중인 런을 평가하지 않도록)
    if (( !DRY )); then
      n=$( cd "$ROOT/extract_5_tasks" && python "$INF" "${INF_ARGS[@]}" --list 2>/dev/null | grep -oE "\] [0-9]+ ckpt" | grep -oE "[0-9]+" | tail -1 )
      if [ "${n:-0}" != "11" ]; then
        echo "[ERR] ${RUN}: 인식된 체크포인트가 ${n:-0}개 (11 이어야 함). 학습이 끝나지 않았거나 경로 문제." | tee -a "$LOG"
        ( cd "$ROOT/extract_5_tasks" && python "$INF" "${INF_ARGS[@]}" --list 2>/dev/null | tail -14 ) | tee -a "$LOG"
        FAILED+=("$RUN"); continue
      fi
    fi
    (( DO_5TASK )) && { run_step "5-task 추론 ($INF)" "$ROOT/extract_5_tasks" python "$INF" "${INF_ARGS[@]}" || ok=0; }
    (( DO_BLIND )) && { run_step "blind 추론 ($BEN)" "$ROOT/extract_benchmark" python "$BEN" --run "$RUN" || ok=0; }
  fi
  (( DO_5TASK )) && { run_step "5-task 채점" "$ROOT/extract_5_tasks"  python acc_parsing.py --model "$RUN" || ok=0; }
  (( DO_BLIND )) && { run_step "blind 채점"  "$ROOT/extract_benchmark" python acc_blind.py  --model "${RUN}_blind" || ok=0; }
  (( ok )) || FAILED+=("$RUN")
  (( DRY )) || echo "[EVAL] $(date '+%F %T') END run=${RUN} ok=${ok}" >> "$LOG"
  (( DRY )) || conda deactivate 2>/dev/null || true
done

echo; echo "============================================================"
if (( ${#FAILED[@]} )); then echo "[EVAL] 실패한 런: ${FAILED[*]}"; exit 1; fi
echo "[EVAL] 완료: ${RUNS[*]}   ($(date '+%m-%d %H:%M'))"
echo "  CSV 갱신: python ${ROOT}/tools/export_csv.py"
