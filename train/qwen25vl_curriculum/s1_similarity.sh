#!/bin/bash
set -euo pipefail
IN_MODEL="$1"
OUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ⚠ S1 은 nohex(프롬프트에서 hex 코드 제거) 버전을 쓴다. 학습/검증/추론 전부 nohex 로 통일.
#    hex 프롬프트는 정답을 그대로 흘려서(색 유사도를 hex 숫자로 계산 가능) 이미지를 안 봐도 풀린다.
#    hex 버전으로 돌리려면 S1_DATA / S1_EVAL_DATA 로 덮어쓰면 된다.
DATA_PATH="${S1_DATA:-/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/rebuttal_nohex/similarity_non_flamingo_nohex.json}"
EVAL_DATA_PATH="${S1_EVAL_DATA:-/home/hail/HDD/vlm_validation/S1_Color_Similarity/dataset/rebuttal_nohex/color_similarity_non_flamingo_nohex.json}"
bash "${SCRIPT_DIR}/_stage.sh" "S1_Similarity" "${IN_MODEL}" "${OUT_DIR}" "${DATA_PATH}" "${EVAL_DATA_PATH}"
