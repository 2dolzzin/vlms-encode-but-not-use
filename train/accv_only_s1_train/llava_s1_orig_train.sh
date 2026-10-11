#!/bin/bash
set -euo pipefail

export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800

BASE_MODEL="/home/hail/vlm3/Model/LLaVA/llava-v1.5-7b"
OUT_DIR="/home/hail/vlm3/checkpoint/accv_only_s1_train/llava_orig"
DATA_PATH="/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/similarity_non_flamingo.json"

export WANDB_PROJECT="accv_only_s1_train"

LLAVA_DIR="/home/hail/vlm3/Model/LLaVA"
cd "$LLAVA_DIR"

echo "===================================================="
echo "[ACCV_Only_S1_Orig] LLaVA"
echo "  base_model = ${BASE_MODEL}"
echo "  out_dir    = ${OUT_DIR}"
echo "  data       = ${DATA_PATH}"
echo "===================================================="

mkdir -p "${OUT_DIR}/end"

python -u llava/train/train_mem.py \
    --model_name_or_path "${BASE_MODEL}" \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "/" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
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
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "ACCV_Only_S1_Orig_LLaVA"

mv "${OUT_DIR}/end/checkpoint-"* "${OUT_DIR}/" 2>/dev/null || true
echo "[ACCV_Only_S1_Orig] LLaVA done → ${OUT_DIR}"
