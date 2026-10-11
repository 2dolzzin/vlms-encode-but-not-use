export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800

#!/bin/bash
set -euo pipefail

BASE_MODEL="/home/hail/vlm3/Model/llava-phi/Mipha-3B-phi2"
OUT_ROOT="/home/hail/vlm3/checkpoint/Mipha_2"

export WANDB_PROJECT="mipha_curriculum2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd /home/hail/vlm3/Model/llava-phi

echo "============================================"
echo "[Curriculum] START"
echo "  base_model = ${BASE_MODEL}"
echo "  out_root   = ${OUT_ROOT}"
echo "============================================"

bash "${SCRIPT_DIR}/s0_logmar.sh"     "${BASE_MODEL}"                        "${OUT_ROOT}/S0_LogMAR-phi2"
bash "${SCRIPT_DIR}/s1_similarity.sh" "${OUT_ROOT}/S0_LogMAR-phi2/end"      "${OUT_ROOT}/S1_Similarity-phi2"
bash "${SCRIPT_DIR}/s2_ishihara.sh"   "${OUT_ROOT}/S1_Similarity-phi2/end"  "${OUT_ROOT}/S2_Ishihara-phi2"
bash "${SCRIPT_DIR}/s3_naming.sh"     "${OUT_ROOT}/S2_Ishihara-phi2/end"    "${OUT_ROOT}/S3_Naming-phi2"
bash "${SCRIPT_DIR}/s4_depth.sh"      "${OUT_ROOT}/S3_Naming-phi2/end"      "${OUT_ROOT}/S4_Depth-phi2"

echo "============================================"
echo "[Curriculum] DONE"
echo "============================================"
