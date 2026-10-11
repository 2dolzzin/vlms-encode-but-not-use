#!/bin/bash
# OpenFlamingo-3B 임의 순서 커리큘럼 학습 (다른 모델의 run_order 와 같은 방식).
#   CUDA_VISIBLE_DEVICES=0 bash run_order.sh 1 4 2 0 3
# → checkpoint/OpenFlamingo_order_14203_nohex/{1_S1_Similarity, 2_S4_Depth, ... }
#   체이닝은 flamingo_model.pt (stage 디렉토리 안) 로 이어진다.
#   S1 은 기본 nohex(similarity_flamingo_nohex.json). hex 로 돌리려면 S1_NOHEX=0
#   START_POS=N 이면 N 번째 스테이지부터 재개.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: CUDA_VISIBLE_DEVICES=<id> $0 <stage idx ...>   e.g. $0 1 4 2 0 3" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29620}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# run_all.sh 와 동일한 학습 설정
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/hail/vlm3/Model/open_flamingo/ckpts/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt}"
export TRAIN_MODE="${TRAIN_MODE:-adapter}"
export FREEZE_LM_EMB="${FREEZE_LM_EMB:-True}"
export SEED="${SEED:-0}"
export PER_DEV_BSZ="${PER_DEV_BSZ:-16}"
export GRAD_ACC="${GRAD_ACC:-1}"
LR="${LR:-1e-6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGE_KEY=(s0 s1 s2 s3 s4)
STAGE_NAME=(S0_LogMAR S1_Similarity S2_Ishihara S3_Naming S4_Depth)

ORDER=("$@")
for i in "${ORDER[@]}"; do
  [[ "$i" =~ ^[0-4]$ ]] || { echo "invalid stage idx: $i (0~4)" >&2; exit 1; }
done

TAG="$(printf '%s' "${ORDER[@]}")"

# S1 데이터: 기본 nohex (OpenFlamingo 용 flamingo 포맷 nohex)
S1_NOHEX="${S1_NOHEX:-1}"
if [ "${S1_NOHEX}" = "1" ]; then
  export S1_DATA="/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/rebuttal_nohex/similarity_flamingo_nohex.json"
  TAG="${TAG}_nohex"
else
  unset S1_DATA
fi
export WANDB_PROJECT="openflamingo_curriculum_order_${TAG}"
OUT_ROOT="/home/hail/vlm3/checkpoint/OpenFlamingo_order_${TAG}"

[[ -f "${PRETRAINED_CKPT}" ]] || { echo "[ERROR] pretrained ckpt not found: ${PRETRAINED_CKPT}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}"
{ printf '%s\n' "${ORDER[@]/#/S}"; echo "s1_data=${S1_DATA:-flamingo(hex, default)}"; } > "${OUT_ROOT}/order.txt"

echo "============================================"
echo "[OF Curriculum order=${ORDER[*]}] START"
echo "  pretrained = ${PRETRAINED_CKPT}"
echo "  out_root   = ${OUT_ROOT}"
echo "  gpu        = ${CUDA_VISIBLE_DEVICES}  (master_port=${MASTER_PORT})"
echo "  mode       = ${TRAIN_MODE}  batch=${PER_DEV_BSZ}x${GRAD_ACC}  lr=${LR}  seed=${SEED}"
echo "  s1_data    = ${S1_DATA:-flamingo(hex, default)}"
echo "============================================"

START_POS="${START_POS:-1}"
prev="${PRETRAINED_CKPT}"
if (( START_POS > 1 )); then
  pi=$(( START_POS - 2 ))
  prev="${OUT_ROOT}/$(( START_POS - 1 ))_${STAGE_NAME[${ORDER[$pi]}]}/flamingo_model.pt"
  [ -f "${prev}" ] || { echo "[ERR] 재개 입력이 없음: ${prev}" >&2; exit 1; }
  echo "  resume     : ${START_POS} 단계부터, 입력 = ${prev}"
fi

pos=1
for i in "${ORDER[@]}"; do
  out_dir="${OUT_ROOT}/${pos}_${STAGE_NAME[$i]}"
  if (( pos < START_POS )); then
    echo "[skip] ${pos}_${STAGE_NAME[$i]} (이미 완료)"
    pos=$((pos + 1)); continue
  fi
  # wandb run 이름이 스테이지/순서마다 유일하도록
  WANDB_NAME="${STAGE_NAME[$i]}_${TAG}" \
    bash "${SCRIPT_DIR}/train.sh" "${STAGE_KEY[$i]}" "${prev}" "${out_dir}" "${LR}"
  prev="${out_dir}/flamingo_model.pt"
  pos=$((pos + 1))
done

echo "============================================"
echo "[OF Curriculum order=${ORDER[*]}] DONE  → ${OUT_ROOT}"
echo "============================================"
