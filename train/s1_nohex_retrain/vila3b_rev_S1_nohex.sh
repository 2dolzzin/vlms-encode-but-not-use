#!/bin/bash
set -euo pipefail

DEFAULT_RUN_NAME="S1_Similarity_nohex_rev"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=16
DEFAULT_GRADIENT_ACCUMULATION_STEPS=16

IN_MODEL="/home/hail/vlm3/checkpoint/VILA_3B_reverse/S2_Ishihara/end"
OUT_DIR="/home/hail/vlm3/checkpoint/VILA_3B_reverse/S1_Similarity_nohex"

export GRADIENT_CHECKPOINTING="False"
export DEEPSPEED_CONFIG="scripts/zero2.json"
export CONV_VERSION="vicuna_v1"

echo "===================================================="
echo "[S1_Similarity_nohex / VILA-3B / reverse]"
echo "  in_model  = ${IN_MODEL}"
echo "  out_dir   = ${OUT_DIR}"
echo "  data_mixture = s1_similarity_nohex"
echo "===================================================="

cd /home/hail/vlm3/Model/VILA

mkdir -p "${OUT_DIR}/end"
OUTPUT_DIR="${OUT_DIR}/end"
source /home/hail/vlm3/train/vila_curriculum/train.sh

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed ${DEEPSPEED_CONFIG} \
        --run_name "S1_Similarity_nohex_rev" \
        --model_name_or_path "${IN_MODEL}" \
        --version "${CONV_VERSION}" \
        --data_mixture s1_similarity_nohex \
        --image_folder "/" \
        --vision_tower "${IN_MODEL}/vision_tower" \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --mm_vision_select_layer -2 \
        --tune_vision_tower False \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length False \
        --vflan_no_system_prompt True \
        --bf16 True \
        --tf32 True \
        --output_dir "${OUT_DIR}/end" \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 1563 \
        --save_total_limit 2 \
        --learning_rate 1e-6 \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --model_max_length 2048 \
        --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
        --dataloader_num_workers 0 \
        --lazy_preprocess True \
        --report_to wandb

mv "${OUT_DIR}/end/checkpoint-"* "${OUT_DIR}/"
find "${OUT_DIR}" -maxdepth 1 -name "checkpoint-*" | sort -t- -k2 -n | tail -1 | xargs rm -rf
echo "[S1_Similarity_nohex / VILA-3B / reverse] done → ${OUT_DIR}"
