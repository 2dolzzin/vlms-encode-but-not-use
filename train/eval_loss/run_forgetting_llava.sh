set -uo pipefail

LLAVA_DIR="/home/hail/vlm3/Model/LLaVA"
HERE="/home/hail/vlm3/train/eval_loss"
WRAP="${HERE}/eval_llava.py"
RESULTS="${HERE}/results_llava"
TMP_OUT="${HERE}/_tmp_out_llava"
mkdir -p "${RESULTS}" "${TMP_OUT}"

DUMMY_DATA="/home/hail/HDD/vlm_training/S0_Logmar/dataset/logmar_non_flamingo.json"

export EVAL_SETS_JSON='[
  ["S0","/home/hail/HDD/vlm_validation/S0_Logmar/dataset/logmar_non_flamingo.json"],
  ["S1","/home/hail/HDD/vlm_validation/S1_Color_Similarity/dataset/color_similarity_non_flamingo.json"],
  ["S2","/home/hail/HDD/vlm_validation/S2_Ishihara/dataset/ishihara_non_flamingo.json"],
  ["S3","/home/hail/HDD/vlm_validation/S3_Color_Naming/dataset/color_naming_non_flamingo.json"],
  ["S4","/home/hail/HDD/vlm_validation/S4_Depth/dataset/depth_non_flamingo.json"]
]'

BASE_MODEL="/home/hail/vlm3/Model/LLaVA/llava-v1.5-7b"
FWD_ROOT="/home/hail/vlm3/checkpoint/llava"
REV_ROOT="/home/hail/vlm3/checkpoint/llava_reverse"

FWD_STATES=(
  "start|${BASE_MODEL}"
  "S0_mid|${FWD_ROOT}/S0_LogMAR/checkpoint-1563"
  "S0_end|${FWD_ROOT}/S0_LogMAR/end"
  "S1_mid|${FWD_ROOT}/S1_Similarity/checkpoint-1563"
  "S1_end|${FWD_ROOT}/S1_Similarity/end"
  "S2_mid|${FWD_ROOT}/S2_Ishihara/checkpoint-1563"
  "S2_end|${FWD_ROOT}/S2_Ishihara/end"
  "S3_mid|${FWD_ROOT}/S3_Naming/checkpoint-1563"
  "S3_end|${FWD_ROOT}/S3_Naming/end"
  "S4_mid|${FWD_ROOT}/S4_Depth/checkpoint-1563"
  "S4_end|${FWD_ROOT}/S4_Depth/end"
)

REV_STATES=(
  "start|${BASE_MODEL}"
  "S4_mid|${REV_ROOT}/S4_Depth/checkpoint-1563"
  "S4_end|${REV_ROOT}/S4_Depth/end"
  "S3_mid|${REV_ROOT}/S3_Naming/checkpoint-1563"
  "S3_end|${REV_ROOT}/S3_Naming/end"
  "S2_mid|${REV_ROOT}/S2_Ishihara/checkpoint-1563"
  "S2_end|${REV_ROOT}/S2_Ishihara/end"
  "S1_mid|${REV_ROOT}/S1_Similarity/checkpoint-1563"
  "S1_end|${REV_ROOT}/S1_Similarity/end"
  "S0_mid|${REV_ROOT}/S0_LogMAR/checkpoint-1563"
  "S0_end|${REV_ROOT}/S0_LogMAR/end"
)

run_one () {
  local dir="$1" tag="$2" model="$3"
  local out_json="${RESULTS}/${RUN_KEY:-$dir}__${tag}.json"
  if [[ -f "${out_json}" ]]; then
    echo "[skip] ${dir}/${tag} (이미 있음)"
    return 0
  fi
  if [[ ! -e "${model}" ]]; then
    echo "[WARN] 모델 경로 없음, 건너뜀: ${model}"
    return 0
  fi
  echo "===================================================="
  echo "[run] ${dir}/${tag}"
  echo "  model = ${model}"
  echo "  out   = ${out_json}"
  echo "===================================================="
  export EVAL_OUT_JSON="${out_json}"
  export EVAL_MODEL_TAG="${RUN_KEY:-$dir}__${tag}"

  ( cd "${LLAVA_DIR}" && python -u "${WRAP}" \
      --model_name_or_path "${model}" \
      --version v1 \
      --data_path "${DUMMY_DATA}" \
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
      --output_dir "${TMP_OUT}" \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --evaluation_strategy "no" \
      --save_strategy "no" \
      --learning_rate 1e-6 \
      --weight_decay 0.0 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --logging_steps 10 \
      --model_max_length 2048 \
      --gradient_checkpointing False \
      --dataloader_num_workers 8 \
      --lazy_preprocess True \
      --report_to none \
      --run_name "forget_${dir}_${tag}" )
  local rc=$?
  if [[ ! -f "${out_json}" ]]; then
    echo "[ERROR] 결과 JSON 미생성 (${dir}/${tag}), rc=${rc}"
  fi
}


# ── 무작위 순서 런 (RUN_KEY=<model>_order_<tag>, ORDER_ROOT=체크포인트 루트): 상태 목록을 폴더에서 자동 구성, S1 검증셋은 nohex ──
if [[ -n "${RUN_KEY:-}" ]]; then
  : "${ORDER_ROOT:?RUN_KEY 사용 시 ORDER_ROOT 필요}"
  mapfile -t ORDER_STATES < <(python3 /home/hail/vlm3/order_runs.py states "${ORDER_ROOT}" --base "${BASE_MODEL}" --mid "checkpoint-1563" --end "end")
  export EVAL_SETS_JSON="$(S1_NOHEX_VAL="/home/hail/HDD/vlm_validation/S1_Color_Similarity/dataset/rebuttal_nohex/color_similarity_non_flamingo_nohex.json" python3 -c 'import json,os; s=json.loads(os.environ["EVAL_SETS_JSON"]); print(json.dumps([[k, os.environ["S1_NOHEX_VAL"] if k=="S1" else p] for k,p in s]))')"
  DIRS="order"
  echo "[order] ${RUN_KEY}: ${#ORDER_STATES[@]} 상태, root=${ORDER_ROOT}"
fi
DIRS="${DIRS:-fwd rev}"
for d in ${DIRS}; do
  if [[ "$d" == "fwd" ]]; then states=("${FWD_STATES[@]}"); elif [[ "$d" == "order" ]]; then states=("${ORDER_STATES[@]}"); else states=("${REV_STATES[@]}"); fi
  for line in "${states[@]}"; do
    tag="${line%%|*}"; model="${line#*|}"
    run_one "$d" "$tag" "$model"
  done
done

echo "[done] 결과 → ${RESULTS}/"
