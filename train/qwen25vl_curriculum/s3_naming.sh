#!/bin/bash
set -euo pipefail
IN_MODEL="$1"
OUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${S3_DATA:-/home/hail/HDD/vlm_training/S3_Color_Naming/dataset/color_naming_non_flamingo.json}"
EVAL_DATA_PATH="/home/hail/HDD/vlm_validation/S3_Color_Naming/dataset/color_naming_non_flamingo.json"
bash "${SCRIPT_DIR}/_stage.sh" "S3_Naming" "${IN_MODEL}" "${OUT_DIR}" "${DATA_PATH}" "${EVAL_DATA_PATH}"
