#!/bin/bash
set -euo pipefail
IN_MODEL="$1"
OUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${S2_DATA:-/home/hail/HDD/vlm_training/S2_Ishihara/dataset/ishihara_non_flamingo.json}"
EVAL_DATA_PATH="/home/hail/HDD/vlm_validation/S2_Ishihara/dataset/ishihara_non_flamingo.json"
bash "${SCRIPT_DIR}/_stage.sh" "S2_Ishihara" "${IN_MODEL}" "${OUT_DIR}" "${DATA_PATH}" "${EVAL_DATA_PATH}"
