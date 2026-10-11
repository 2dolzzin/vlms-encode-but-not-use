#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "[ACCV_Only_S1] START"
echo "============================================"

bash "${SCRIPT_DIR}/llava_s1_train.sh"
bash "${SCRIPT_DIR}/mipha_s1_train.sh"
bash "${SCRIPT_DIR}/vila_3b_s1_train.sh"
bash "${SCRIPT_DIR}/vila_8b_s1_train.sh"

echo "============================================"
echo "[ACCV_Only_S1] DONE"
echo "============================================"
