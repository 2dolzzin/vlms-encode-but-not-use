#!/bin/bash
set -euo pipefail

export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800

BASE_MODEL="/home/hail/vlm3/Model/llava-phi/Mipha-3B-phi2"
OUT_DIR="/home/hail/vlm3/checkpoint/accv_only_s1_train/mipha"
DATA_PATH="/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/similarity_non_flamingo_3.json"

export WANDB_PROJECT="accv_only_s1_train"

cd /home/hail/vlm3/Model/llava-phi

echo "===================================================="
echo "[ACCV_Only_S1] Mipha"
echo "  base_model = ${BASE_MODEL}"
echo "  out_dir    = ${OUT_DIR}"
echo "  data       = ${DATA_PATH}"
echo "===================================================="

mkdir -p "${OUT_DIR}/end"

deepspeed --master_port 29600 /home/hail/vlm3/train/mipha_curriculum/train_with_eval.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${BASE_MODEL}" \
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
    --run_name "ACCV_Only_S1_Mipha"

mv "${OUT_DIR}/end/checkpoint-"* "${OUT_DIR}/"
cp "${BASE_MODEL}/preprocessor_config.json" "${OUT_DIR}/end/"
cp "${BASE_MODEL}/preprocessor_config.json" "${OUT_DIR}/checkpoint-"*/
echo "[ACCV_Only_S1] Mipha done → ${OUT_DIR}"
