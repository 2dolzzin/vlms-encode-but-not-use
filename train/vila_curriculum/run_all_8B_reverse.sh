#!/bin/bash
set -euo pipefail

BASE_MODEL="/home/hail/vlm3/Model/VILA/Llama-3-VILA1.5-8B"
OUT_ROOT="/home/hail/vlm3/checkpoint/VILA_8B_reverse"

export WANDB_PROJECT="vila_8b_curriculum_reverse"
export GLOBAL_TRAIN_BATCH_SIZE=16
export GRADIENT_ACCUMULATION_STEPS=4
export GRADIENT_CHECKPOINTING="True"
export DEEPSPEED_CONFIG="scripts/zero3_offload.json"
export CONV_VERSION="llama_3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd /home/hail/vlm3/Model/VILA

echo "==========================================="
echo "[Reverse Curriculum] START"
echo "  base_model = ${BASE_MODEL}"
echo "  out_root   = ${OUT_ROOT}"
echo "============================================"

bash "${SCRIPT_DIR}/s4_depth.sh"      "${BASE_MODEL}"                         "${OUT_ROOT}/S4_Depth"
bash "${SCRIPT_DIR}/s3_naming.sh"     "${OUT_ROOT}/S4_Depth/end"              "${OUT_ROOT}/S3_Naming"
bash "${SCRIPT_DIR}/s2_ishihara.sh"   "${OUT_ROOT}/S3_Naming/end"             "${OUT_ROOT}/S2_Ishihara"
bash "${SCRIPT_DIR}/s1_similarity.sh" "${OUT_ROOT}/S2_Ishihara/end"           "${OUT_ROOT}/S1_Similarity"
bash "${SCRIPT_DIR}/s0_logmar.sh"     "${OUT_ROOT}/S1_Similarity/end"         "${OUT_ROOT}/S0_LogMAR"
echo "============================================"
echo "[Reverse Curriculum] DONE"

echo "============================================"
