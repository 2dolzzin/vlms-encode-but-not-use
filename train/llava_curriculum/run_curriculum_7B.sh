set -euo pipefail

export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800

BASE_MODEL="/home/hail/vlm3/Model/LLaVA/llava-v1.5-7b"
LLAVA_DIR="/home/hail/vlm3/Model/LLaVA"

declare -A DATA=(
  [S0_LogMAR]="/home/hail/HDD/vlm_training/S0_Logmar/dataset/logmar_non_flamingo.json"
  [S1_Similarity]="/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/similarity_non_flamingo.json"
  [S2_Ishihara]="/home/hail/HDD/vlm_training/S2_Ishihara/dataset/ishihara_non_flamingo.json"
  [S3_Naming]="/home/hail/HDD/vlm_training/S3_Color_Naming/dataset/color_naming_non_flamingo.json"
  [S4_Depth]="/home/hail/HDD/vlm_training/S4_Depth/dataset/depth_non_flamingo.json"
)

  ORDER=(S0_LogMAR S1_Similarity S2_Ishihara S3_Naming S4_Depth)
  OUT_ROOT="/home/hail/vlm3/checkpoint/llava"

  export WANDB_PROJECT="llava_curriculum"
  TITLE="Curriculum"

train_one() {
  local stage="$1" in_model="$2" out_dir="$3"
  local data_path="${DATA[$stage]}"

  echo "===================================================="
  echo "[$stage]"
  echo "  in_model  = ${in_model}"
  echo "  out_dir   = ${out_dir}"
  echo "  data      = ${data_path}"
  echo "===================================================="

  mkdir -p "${out_dir}/end"

  python -u llava/train/train_mem.py \
      --model_name_or_path "${in_model}" \
      --version v1 \
      --data_path "${data_path}" \
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
      --output_dir "${out_dir}/end" \
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
      --run_name "${stage}"

  mv "${out_dir}/end/checkpoint-"* "${out_dir}/" 2>/dev/null || true
  echo "[$stage] done → ${out_dir}"
}

# ── 실행: end 출력을 다음 스테이지 입력으로 체이닝 ──────────────────────────
cd "$LLAVA_DIR"

echo "============================================"
echo "[$TITLE] START"
echo "  base_model = ${BASE_MODEL}"
echo "  out_root   = ${OUT_ROOT}"
echo "  order      = ${ORDER[*]}"
echo "============================================"

prev="$BASE_MODEL"
for stage in "${ORDER[@]}"; do
  train_one "$stage" "$prev" "${OUT_ROOT}/${stage}"
  prev="${OUT_ROOT}/${stage}/end"
done

echo "============================================"
echo "[$TITLE] DONE"
echo "============================================"
