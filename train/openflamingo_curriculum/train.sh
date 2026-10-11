#!/bin/bash
# 단일 stage 학습기. stage 키만 다르고 나머지는 전부 공통.
# 사용: bash train.sh <STAGE_KEY> <IN_CKPT.pt> <OUT_DIR> [LR]
#   STAGE_KEY ∈ {s0, s1, s2, s3, s4}
set -euo pipefail

STAGE_KEY="$1"
IN_CKPT="$2"
OUT_DIR="$3"
LR="${4:-1e-6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OF_PKG="/home/hail/vlm3/Model/open_flamingo"          # `from open_flamingo import ...`
export PYTHONPATH="${OF_PKG}:${SCRIPT_DIR}:${PYTHONPATH:-}"

# ── stage 키 → (표시이름, 학습 json) ──
case "${STAGE_KEY}" in
  s0) STAGE="S0_LogMAR";    DATA_PATH="/home/hail/HDD/vlm_training/S0_Logmar/dataset/logmar_flamingo.json" ;;
  # ⚠ S1 은 nohex 가 표준(hex 프롬프트는 정답 누출). 기본값은 기존 hex 유지,
  #    nohex 로 돌리려면 S1_DATA 로 덮어쓴다 (run_order.sh 가 지정).
  s1) STAGE="S1_Similarity";DATA_PATH="${S1_DATA:-/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/similarity_flamingo.json}" ;;
  s2) STAGE="S2_Ishihara";  DATA_PATH="/home/hail/HDD/vlm_training/S2_Ishihara/dataset/ishihara_flamingo.json" ;;
  s3) STAGE="S3_Naming";    DATA_PATH="/home/hail/HDD/vlm_training/S3_Color_Naming/dataset/color_naming_flamingo.json" ;;
  s4) STAGE="S4_Depth";     DATA_PATH="/home/hail/HDD/vlm_training/S4_Depth/dataset/depth_flamingo.json" ;;
  *)  echo "[ERROR] unknown STAGE_KEY: ${STAGE_KEY} (s0~s4)" >&2; exit 1 ;;
esac

# ── 공통 하이퍼파라미터 (Mipha/LLaVA/VILA 와 동일, effective batch = 1×16 = 16) ──
PER_DEV_BSZ="${PER_DEV_BSZ:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
MODEL_MAX_LEN="${MODEL_MAX_LEN:-2048}"
SAVE_STEPS="${SAVE_STEPS:-1562}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
SEED="${SEED:-0}"
TRAIN_MODE="${TRAIN_MODE:-adapter}"      # adapter(=perceiver+cross-attn 2개만) / fft
FREEZE_LM_EMB="${FREEZE_LM_EMB:-True}"   # True=임베딩 동결 → 딱 2개 모듈만

[[ -f "${DATA_PATH}" ]] || { echo "[ERROR] train json not found: ${DATA_PATH}" >&2; exit 1; }
[[ -f "${IN_CKPT}"   ]] || { echo "[ERROR] in_ckpt not found: ${IN_CKPT}" >&2; exit 1; }
mkdir -p "${OUT_DIR}"

echo "===================================================="
echo "[OF-Curriculum] ${STAGE}  (train_mode=${TRAIN_MODE}, lr=${LR})"
echo "  in_ckpt = ${IN_CKPT}"
echo "  out_dir = ${OUT_DIR}"
echo "  data    = ${DATA_PATH}"
echo "  batch   = ${PER_DEV_BSZ} x ${GRAD_ACC} | save_steps=${SAVE_STEPS} | seed=${SEED}"
echo "===================================================="

deepspeed --include "localhost:${CUDA_VISIBLE_DEVICES:-0}" --master_port "${MASTER_PORT:-29610}" \
    "${SCRIPT_DIR}/train.py" \
    --deepspeed "${SCRIPT_DIR}/scripts/zero2.json" \
    --vision_encoder_path "ViT-L-14" \
    --vision_encoder_pretrained "openai" \
    --lm_path "anas-awadalla/mpt-1b-redpajama-200b" \
    --tokenizer_path "anas-awadalla/mpt-1b-redpajama-200b" \
    --cross_attn_every_n_layers 1 \
    --train_mode "${TRAIN_MODE}" \
    --freeze_vision_encoder True \
    --freeze_lm_embeddings "${FREEZE_LM_EMB}" \
    --pretrained_checkpoint "${IN_CKPT}" \
    --data_path "${DATA_PATH}" \
    --image_folder "/" \
    --image_aspect_ratio "pad" \
    --lazy_preprocess True \
    --bf16 True \
    --tf32 True \
    --output_dir "${OUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEV_BSZ}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 3 \
    --save_only_model True \
    --save_safetensors False \
    --learning_rate "${LR}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length "${MODEL_MAX_LEN}" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to "wandb" \
    --seed "${SEED}" \
    --run_name "${WANDB_NAME:-${STAGE}_lr${LR}}"

echo "[OF-Curriculum] ${STAGE} done → ${OUT_DIR}/flamingo_model.pt"
