#!/usr/bin/env bash
# run_probing.sh — 런(run 키) 단위로 프로빙 배터리 일괄 실행. 임베딩은 추론 단계에서 이미 저장돼 있어야 함.
#
#   CUDA_VISIBLE_DEVICES=1 bash run_probing.sh llava_order_14203 qwen25vl_7b qwen25vl_7b_reverse
#   CUDA_VISIBLE_DEVICES=1 bash run_probing.sh --with-logreg llava_order_14203    # MLP/ridge + 선형 logreg 까지
#   bash run_probing.sh --logreg-only llava_order_14203 qwen25vl_7b              # sklearn 선형 logreg 만 (CPU 전용, GPU 불필요)
#   CUDA_VISIBLE_DEVICES=1 bash run_probing.sh --linear llava_order_14203         # GPU 선형 프로브(torch, 6지점, MLP 와 같은 프로토콜) 만
#   CUDA_VISIBLE_DEVICES=1 bash run_probing.sh --with-linear llava_order_14203    # MLP/ridge + GPU 선형
#   bash run_probing.sh --dry llava_order_14203
#
# 런당 수행 (본 분석 = 논문 Exp 2/4 설정):
#   inf  : probe_classifier (MLP512, 이산 4과제)          → probing/<run>/inf_classification_{1,2,3}/
#          probe_regressor  (RidgeCV, S1_similarity)      → probing/<run>/inf_regression_{1,2,3}/ridge/
#   blind: probe_classifier (MLP512, blind 8 + line_plot + mmvp) → probing/<run>_blind/ben_classification_{1,2,3}/
#   linear (--with-linear / --linear): probe_classifier --probe linear = 은닉층 없는 softmax 선형, GPU, MLP 와 완전히 같은 분할·시드·early-stopping
#          → probing/<run>/inf_classification_linear_{1,2,3}/, probing/<run>_blind/ben_classification_linear_{1,2,3}/  (런당 ~40분)
#   logreg (--with-logreg / --logreg-only): probe_logreg = 선형 multinomial logistic, CPU 프로세스 풀(JOBS, 기본 16)
#          inf  → probing/<run>/inf_classification_logreg_{1,2,3}/
#          blind→ probing/<run>_blind/ben_classification_logreg_{1,2,3}/
#          기본은 기존 fwd/rev 결과(rebuttal Exp A)와 같은 projector 지점만. LM 레이어까지 6지점 전부는 --logreg-all-layers
#          (proj-only ≈ 1h/런, all-layers ≈ 5~6배).
#          GPU 를 안 쓰므로 MLP 프로빙이 GPU 에서 도는 동안 CPU 에서 동시에 돌려도 된다.
# 임베딩 위치: extract_5_tasks/result/<run>/embedding, extract_benchmark/result/<run>_blind/embeddings
# 결과 json 이 이미 있는 (ckpt, task) 는 건너뛰므로(PROBE_SKIP_EXISTING=1) 중간에 죽어도 같은 명령으로 이어서 돈다.
# 로그: probing/logs/<run>.log
set -o pipefail
cd "$(dirname "$0")"
mkdir -p logs
export PROBE_SKIP_EXISTING=1
export PYTHONUNBUFFERED=1
# 한 GPU 에 여러 프로세스를 동시에 올릴 때 코어 과점유 방지 (프로세스당 ~3코어). 바꾸려면 OMP_NUM_THREADS=... 로.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
PY=/home/hail/anaconda3/envs/llava/bin/python   # torch+sklearn+scipy+wandb 모두 있음 (run_s1_regression.sh 와 동일)
JOBS="${JOBS:-16}"

WITH_LOGREG=0; LOGREG_ONLY=0; LOGREG_ALL=0; WITH_LINEAR=0; LINEAR_ONLY=0; DRY=0; RUNS=()
for a in "$@"; do
  case "$a" in
    --with-logreg) WITH_LOGREG=1 ;;
    --logreg-only) WITH_LOGREG=1; LOGREG_ONLY=1 ;;
    --logreg-all-layers) LOGREG_ALL=1 ;;
    --with-linear) WITH_LINEAR=1 ;;
    --linear) WITH_LINEAR=1; LINEAR_ONLY=1 ;;
    --dry) DRY=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    -*) echo "[ERR] 알 수 없는 옵션: $a" >&2; exit 1 ;;
    *) RUNS+=("$a") ;;
  esac
done
[ "${#RUNS[@]}" -ge 1 ] || { echo "usage: CUDA_VISIBLE_DEVICES=<id> $0 [--with-logreg] [--dry] <run_key> [...]" >&2; exit 1; }
if (( DRY == 0 && LOGREG_ONLY == 0 )); then   # MLP 또는 GPU 선형이 포함되면 GPU 필요
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "[ERR] CUDA_VISIBLE_DEVICES 를 지정하세요 (MLP 프로브가 GPU 를 씁니다; 선형만이면 --logreg-only)" >&2; exit 1; }
fi
(( LOGREG_ONLY )) && export CUDA_VISIBLE_DEVICES=""   # 선형 프로브는 CPU 전용 — GPU 를 잡지 않게

step() {   # step <라벨> <명령...>
  local label="$1"; shift
  echo; echo "──── [$label] $(date '+%m-%d %H:%M') ────"; echo "  $*"
  (( DRY )) && return 0
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  (( rc == 0 )) || { echo "[FAIL] $label (exit $rc)" | tee -a "$LOG"; return $rc; }
}

FAILED=()
for RUN in "${RUNS[@]}"; do
  LOG="logs/${RUN}.log"
  INF_EMB="../extract_5_tasks/result/${RUN}/embedding"; BEN_EMB="../extract_benchmark/result/${RUN}_blind/embeddings"
  n_inf=$(find "$INF_EMB" -name '*.pt' 2>/dev/null | wc -l); n_ben=$(find "$BEN_EMB" -name '*.pt' 2>/dev/null | wc -l)
  echo "============================================================"
  echo "[PROBE] run=${RUN}  gpu=${CUDA_VISIBLE_DEVICES:-none}  mlp=$(( (1-LOGREG_ONLY)*(1-LINEAR_ONLY) )) linear_gpu=${WITH_LINEAR} logreg=${WITH_LOGREG}($([ $LOGREG_ALL = 1 ] && echo 6지점 || echo proj-only), jobs=${JOBS})  임베딩: inf=${n_inf}/55  blind=${n_ben}/110"
  echo "============================================================"
  (( DRY )) || echo "[PROBE] $(date '+%F %T') START run=${RUN} inf=${n_inf} blind=${n_ben} logreg=${WITH_LOGREG}" >> "$LOG"
  if (( n_inf < 55 && DRY == 0 )); then echo "[WARN] ${RUN}: inf 임베딩이 ${n_inf}/55 — 추론이 덜 끝났으면 먼저 run_eval.sh 를 완료하세요 (있는 것만 프로빙)" | tee -a "$LOG"; fi
  if (( n_ben < 110 && DRY == 0 )); then echo "[WARN] ${RUN}: blind 임베딩이 ${n_ben}/110" | tee -a "$LOG"; fi
  ok=1
  if (( LOGREG_ONLY == 0 && LINEAR_ONLY == 0 )); then
    step "inf MLP classifier"       "$PY" probe_classifier.py --model "$RUN" --source inf || ok=0
    step "inf ridge S1_similarity"  "$PY" probe_regressor.py  --model "$RUN" --source inf --task S1_similarity --probe ridge || ok=0
    step "blind MLP classifier"     "$PY" probe_classifier.py --model "${RUN}_blind" --source ben || ok=0
  fi
  if (( WITH_LINEAR )); then
    step "inf linear probe (GPU, 6지점)"   "$PY" probe_classifier.py --model "$RUN"         --source inf --probe linear || ok=0
    step "blind linear probe (GPU, 6지점)" "$PY" probe_classifier.py --model "${RUN}_blind" --source ben --probe linear || ok=0
  fi
  if (( WITH_LOGREG )); then
    if (( LOGREG_ALL )); then
      step "inf logreg (선형, 6지점 전부)"   "$PY" probe_logreg.py --model "$RUN"         --source inf --jobs "$JOBS" || ok=0
      step "blind logreg (선형, 6지점 전부)" "$PY" probe_logreg.py --model "${RUN}_blind" --source ben --jobs "$JOBS" || ok=0
    else
      step "inf logreg (선형, proj 만)"   "$PY" probe_logreg.py --model "$RUN"         --source inf --proj_only --jobs "$JOBS" || ok=0
      step "blind logreg (선형, proj 만)" "$PY" probe_logreg.py --model "${RUN}_blind" --source ben --proj_only --jobs "$JOBS" || ok=0
    fi
  fi
  (( ok )) || FAILED+=("$RUN")
  (( DRY )) || echo "[PROBE] $(date '+%F %T') END run=${RUN} ok=${ok}" >> "$LOG"
done
echo; echo "============================================================"
if (( ${#FAILED[@]} )); then echo "[PROBE] 실패한 런: ${FAILED[*]}"; exit 1; fi
echo "[PROBE] 완료: ${RUNS[*]}   ($(date '+%m-%d %H:%M'))"
echo "  표 갱신: python ../tools/export_csv.py && python ../tools/make_tables.py"
