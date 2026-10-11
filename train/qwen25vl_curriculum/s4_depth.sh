#!/bin/bash
set -euo pipefail
IN_MODEL="$1"
OUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${S4_DATA:-/home/hail/HDD/vlm_training/S4_Depth/dataset/depth_non_flamingo.json}"
EVAL_DATA_PATH="/home/hail/HDD/vlm_validation/S4_Depth/dataset/depth_non_flamingo.json"
bash "${SCRIPT_DIR}/_stage.sh" "S4_Depth" "${IN_MODEL}" "${OUT_DIR}" "${DATA_PATH}" "${EVAL_DATA_PATH}"
