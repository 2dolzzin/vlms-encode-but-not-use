#!/bin/bash
set -euo pipefail

IN_MODEL="$1"
OUT_DIR="$2"

DATA_PATH="/home/hail/HDD/vlm_training/S4_Depth/dataset/depth_non_flamingo.json"
export EVAL_DATA_PATH="/home/hail/HDD/vlm_validation/S4_Depth/dataset/depth_non_flamingo.json"

echo "===================================================="
echo "[S4_Depth]"
echo "  in_model  = ${IN_MODEL}"
echo "  out_dir   = ${OUT_DIR}"
echo "  data      = ${DATA_PATH}"
echo "  eval_data = ${EVAL_DATA_PATH}"
echo "===================================================="

mkdir -p "${OUT_DIR}/end"

deepspeed --include "localhost:${GPU_IDS:-0}" --master_port "${MASTER_PORT:-29600}" /home/hail/vlm3/train/mipha_curriculum/train_with_eval.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${IN_MODEL}" \
    --version v0 \
    --data_path "${DATA_PATH}" \
    --image_folder "/" \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --freeze_backbone False \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --tf32 True \
    --output_dir "${OUT_DIR}/end" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy steps \
    --save_steps 1563 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "S4_Depth${RUN_SUFFIX:-}"

mv "${OUT_DIR}/end/checkpoint-"* "${OUT_DIR}/"
cp "${IN_MODEL}/preprocessor_config.json" "${OUT_DIR}/end/"
cp "${IN_MODEL}/preprocessor_config.json" "${OUT_DIR}/checkpoint-"*/
echo "[S4_Depth] done → ${OUT_DIR}"
