export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800

#!/bin/bash
set -euo pipefail

BASE_MODEL="/home/hail/vlm3/Model/LLaVA/llava-v1.5-7b"
OUT_ROOT="/home/hail/vlm3/checkpoint/llava_2"

export WANDB_PROJECT="llava_curriculum_2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd /home/hail/vlm3/Model/LLaVA

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
