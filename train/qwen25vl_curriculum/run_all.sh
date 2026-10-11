#!/bin/bash
# run_all.sh — Qwen2.5-VL-7B forward 커리큘럼 (S0→S1→S2→S3→S4). llava run_all_7B.sh 와 동일 구조.
#   conda activate qwenvl && bash run_all.sh
#   NUM_GPUS=2 GPU_IDS=0,1 bash run_all.sh      # 기본
set -euo pipefail
export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800

BASE_MODEL="/home/hail/vlm3/Model/Qwen2.5-VL-7B-Instruct"
OUT_ROOT="${OUT_ROOT:-/home/hail/vlm3/checkpoint/Qwen25VL_7B}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen25vl_curriculum}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "[Curriculum] START"
echo "  base_model = ${BASE_MODEL}"
echo "  out_root   = ${OUT_ROOT}"
echo "============================================"

bash "${SCRIPT_DIR}/s0_logmar.sh"     "${BASE_MODEL}"                    "${OUT_ROOT}/S0_LogMAR"
bash "${SCRIPT_DIR}/s1_similarity.sh" "${OUT_ROOT}/S0_LogMAR/end"        "${OUT_ROOT}/S1_Similarity"
bash "${SCRIPT_DIR}/s2_ishihara.sh"   "${OUT_ROOT}/S1_Similarity/end"    "${OUT_ROOT}/S2_Ishihara"
bash "${SCRIPT_DIR}/s3_naming.sh"     "${OUT_ROOT}/S2_Ishihara/end"      "${OUT_ROOT}/S3_Naming"
bash "${SCRIPT_DIR}/s4_depth.sh"      "${OUT_ROOT}/S3_Naming/end"        "${OUT_ROOT}/S4_Depth"

echo "============================================"
echo "[Curriculum] DONE"
echo "============================================"
