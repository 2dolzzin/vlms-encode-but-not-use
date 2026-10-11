#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_PROJECT="${WANDB_PROJECT:-openflamingo_curriculum}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/hail/vlm3/Model/open_flamingo/ckpts/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt}"
OUT_ROOT="${OUT_ROOT:-/home/hail/vlm3/checkpoint/OpenFlamingo}"
export TRAIN_MODE="${TRAIN_MODE:-adapter}"   # adapter(=2개 모듈만) / fft
export FREEZE_LM_EMB="${FREEZE_LM_EMB:-True}"
export SEED="${SEED:-0}"
# effective batch = PER_DEV_BSZ x GRAD_ACC = 16 (기존 1x16과 동일, 단 16개를 병렬 처리해 속도↑)
export PER_DEV_BSZ="${PER_DEV_BSZ:-16}"
export GRAD_ACC="${GRAD_ACC:-1}"

[[ -f "${PRETRAINED_CKPT}" ]] || { echo "[ERROR] pretrained ckpt not found: ${PRETRAINED_CKPT}" >&2; exit 1; }

# 정방향 커리큘럼: s0 -> s4. (stage_key  out_subdir  lr)
STAGES=("s0 S0_LogMAR 1e-6" "s1 S1_Similarity 1e-6" "s2 S2_Ishihara 1e-6" "s3 S3_Naming 1e-6" "s4 S4_Depth 1e-6")

echo "[OF-Curriculum] START forward (s0->s4) | mode=${TRAIN_MODE} | out=${OUT_ROOT}"
PREV_CKPT="${PRETRAINED_CKPT}"
for entry in "${STAGES[@]}"; do
  read -r KEY SUB LR <<< "${entry}"
  OUT_DIR="${OUT_ROOT}/${SUB}"
  bash "${SCRIPT_DIR}/train.sh" "${KEY}" "${PREV_CKPT}" "${OUT_DIR}" "${LR}"
  PREV_CKPT="${OUT_DIR}/flamingo_model.pt"   # 다음 stage는 직전 결과 weight 로드
done
echo "[OF-Curriculum] DONE → ${OUT_ROOT}"
