#!/bin/bash
# run_forgetting_qwen25vl.sh — Qwen2.5-VL-7B 11시점 × (fwd|rev) × 5 held-out 셋 eval_loss.
#   conda activate qwenvl
#   bash run_forgetting_qwen25vl.sh            # fwd + rev
#   DIRS=fwd bash run_forgetting_qwen25vl.sh   # forward 만
#   CUDA_VISIBLE_DEVICES=2 bash run_forgetting_qwen25vl.sh
# 결과: results_qwen25vl_7b/{fwd|rev}__{tag}.json  (results_llava 와 동일 포맷)
set -uo pipefail

HERE="/home/hail/vlm3/train/eval_loss"
RESULTS="${HERE}/results_qwen25vl_7b"
mkdir -p "${RESULTS}"

BASE_MODEL="/home/hail/vlm3/Model/Qwen2.5-VL-7B-Instruct"
FWD_ROOT="/home/hail/vlm3/checkpoint/Qwen25VL_7B"
REV_ROOT="/home/hail/vlm3/checkpoint/Qwen25VL_7B_reverse"

FWD_STATES=(
  "start|${BASE_MODEL}"
  "S0_mid|${FWD_ROOT}/S0_LogMAR/checkpoint-1563"
  "S0_end|${FWD_ROOT}/S0_LogMAR/end"
  "S1_mid|${FWD_ROOT}/S1_Similarity/checkpoint-1563"
  "S1_end|${FWD_ROOT}/S1_Similarity/end"
  "S2_mid|${FWD_ROOT}/S2_Ishihara/checkpoint-1563"
  "S2_end|${FWD_ROOT}/S2_Ishihara/end"
  "S3_mid|${FWD_ROOT}/S3_Naming/checkpoint-1563"
  "S3_end|${FWD_ROOT}/S3_Naming/end"
  "S4_mid|${FWD_ROOT}/S4_Depth/checkpoint-1563"
  "S4_end|${FWD_ROOT}/S4_Depth/end"
)
REV_STATES=(
  "start|${BASE_MODEL}"
  "S4_mid|${REV_ROOT}/S4_Depth/checkpoint-1563"
  "S4_end|${REV_ROOT}/S4_Depth/end"
  "S3_mid|${REV_ROOT}/S3_Naming/checkpoint-1563"
  "S3_end|${REV_ROOT}/S3_Naming/end"
  "S2_mid|${REV_ROOT}/S2_Ishihara/checkpoint-1563"
  "S2_end|${REV_ROOT}/S2_Ishihara/end"
  "S1_mid|${REV_ROOT}/S1_Similarity/checkpoint-1563"
  "S1_end|${REV_ROOT}/S1_Similarity/end"
  "S0_mid|${REV_ROOT}/S0_LogMAR/checkpoint-1563"
  "S0_end|${REV_ROOT}/S0_LogMAR/end"
)

run_one () {
  local dir="$1" tag="$2" model="$3"
  local out_json="${RESULTS}/${RUN_KEY:-$dir}__${tag}.json"
  if [[ -f "${out_json}" ]]; then echo "[skip] ${dir}/${tag} (이미 있음)"; return 0; fi
  if [[ ! -e "${model}" ]]; then echo "[WARN] 모델 경로 없음, 건너뜀: ${model}"; return 0; fi
  echo "===================================================="
  echo "[run] ${dir}/${tag}   model=${model}"
  echo "===================================================="
  python -u "${HERE}/eval_qwen25vl.py" --model "${model}" --tag "${RUN_KEY:-$dir}__${tag}" --out "${out_json}" \
      --batch_size "${EVAL_BS:-8}"
  [[ -f "${out_json}" ]] || echo "[ERROR] 결과 JSON 미생성 (${dir}/${tag})"
}


# ── 무작위 순서 런 (RUN_KEY=<model>_order_<tag>, ORDER_ROOT=체크포인트 루트): 상태 목록을 폴더에서 자동 구성, S1 검증셋은 nohex ──
if [[ -n "${RUN_KEY:-}" ]]; then
  : "${ORDER_ROOT:?RUN_KEY 사용 시 ORDER_ROOT 필요}"
  mapfile -t ORDER_STATES < <(python3 /home/hail/vlm3/order_runs.py states "${ORDER_ROOT}" --base "${BASE_MODEL}" --mid "checkpoint-1563" --end "end")
  DIRS="order"
  echo "[order] ${RUN_KEY}: ${#ORDER_STATES[@]} 상태, root=${ORDER_ROOT}"
fi
DIRS="${DIRS:-fwd rev}"
for d in ${DIRS}; do
  if [[ "$d" == "fwd" ]]; then states=("${FWD_STATES[@]}"); elif [[ "$d" == "order" ]]; then states=("${ORDER_STATES[@]}"); else states=("${REV_STATES[@]}"); fi
  for line in "${states[@]}"; do
    tag="${line%%|*}"; model="${line#*|}"
    run_one "$d" "$tag" "$model"
  done
done
echo "[done] 결과 → ${RESULTS}/"
