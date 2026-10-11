#!/bin/bash
# run_forgetting_mipha.sh
# ─────────────────────────────────────────────────────────────────────────────
# Mipha-3B(phi2) 커리큘럼 체크포인트들의 eval_loss(S0~S4) 측정 런처.
# eval_mipha.py(몽키패치)를 train.py(mipha) 와 "동일한 deepspeed 플래그"로 런치.
# 차이: 스크립트=eval_mipha.py, output_dir=temp, save_strategy=no → 체크포인트 보호.
#
# 사용:
#   bash run_forgetting_mipha.sh            # fwd + rev 전부
#   DIRS=fwd bash run_forgetting_mipha.sh   # 정방향만
#
# 결과: results_mipha/{dir}__{tag}.json = {"model","model_path","losses":{S0..S4}}
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

LLAVA_PHI="/home/hail/vlm3/Model/llava-phi"
HERE="/home/hail/vlm3/train/eval_loss"
WRAP="${HERE}/eval_mipha.py"
RESULTS="${HERE}/results_mipha"
TMP_OUT="${HERE}/_tmp_out_mipha"
mkdir -p "${RESULTS}" "${TMP_OUT}"

# make_supervised_data_module 가 학습셋을 만들 때 쓰는 더미(loss 무관).
DUMMY_DATA="/home/hail/HDD/vlm_training/S0_Logmar/dataset/logmar_non_flamingo.json"

export EVAL_SETS_JSON='[
  ["S0","/home/hail/HDD/vlm_validation/S0_Logmar/dataset/logmar_non_flamingo.json"],
  ["S1","/home/hail/HDD/vlm_validation/S1_Color_Similarity/dataset/color_similarity_non_flamingo.json"],
  ["S2","/home/hail/HDD/vlm_validation/S2_Ishihara/dataset/ishihara_non_flamingo.json"],
  ["S3","/home/hail/HDD/vlm_validation/S3_Color_Naming/dataset/color_naming_non_flamingo.json"],
  ["S4","/home/hail/HDD/vlm_validation/S4_Depth/dataset/depth_non_flamingo.json"]
]'

# ── 평가할 체크포인트 시점들 (디스크 실제: Mipha / Mipha_reverse, dir 은 -phi2) ──
BASE_MODEL="/home/hail/vlm3/Model/llava-phi/Mipha-3B-phi2"
FWD_ROOT="/home/hail/vlm3/checkpoint/Mipha"
REV_ROOT="/home/hail/vlm3/checkpoint/Mipha_reverse"

# "tag|model_path"  (start=base, stage 별 mid=checkpoint-1563 / end)
FWD_STATES=(
  "start|${BASE_MODEL}"
  "S0_mid|${FWD_ROOT}/S0_LogMAR-phi2/checkpoint-1563"
  "S0_end|${FWD_ROOT}/S0_LogMAR-phi2/end"
  "S1_mid|${FWD_ROOT}/S1_Similarity-phi2/checkpoint-1563"
  "S1_end|${FWD_ROOT}/S1_Similarity-phi2/end"
  "S2_mid|${FWD_ROOT}/S2_Ishihara-phi2/checkpoint-1563"
  "S2_end|${FWD_ROOT}/S2_Ishihara-phi2/end"
  "S3_mid|${FWD_ROOT}/S3_Naming-phi2/checkpoint-1563"
  "S3_end|${FWD_ROOT}/S3_Naming-phi2/end"
  "S4_mid|${FWD_ROOT}/S4_Depth-phi2/checkpoint-1563"
  "S4_end|${FWD_ROOT}/S4_Depth-phi2/end"
)
REV_STATES=(
  "start|${BASE_MODEL}"
  "S4_mid|${REV_ROOT}/S4_Depth-phi2/checkpoint-1563"
  "S4_end|${REV_ROOT}/S4_Depth-phi2/end"
  "S3_mid|${REV_ROOT}/S3_Naming-phi2/checkpoint-1563"
  "S3_end|${REV_ROOT}/S3_Naming-phi2/end"
  "S2_mid|${REV_ROOT}/S2_Ishihara-phi2/checkpoint-1563"
  "S2_end|${REV_ROOT}/S2_Ishihara-phi2/end"
  "S1_mid|${REV_ROOT}/S1_Similarity-phi2/checkpoint-1563"
  "S1_end|${REV_ROOT}/S1_Similarity-phi2/end"
  "S0_mid|${REV_ROOT}/S0_LogMAR-phi2/checkpoint-1563"
  "S0_end|${REV_ROOT}/S0_LogMAR-phi2/end"
)

run_one () {
  local dir="$1" tag="$2" model="$3"
  local out_json="${RESULTS}/${RUN_KEY:-$dir}__${tag}.json"
  if [[ -f "${out_json}" ]]; then
    echo "[skip] ${dir}/${tag} (이미 있음)"; return 0
  fi
  if [[ ! -e "${model}" ]]; then
    echo "[WARN] 모델 경로 없음, 건너뜀: ${model}"; return 0
  fi
  echo "===================================================="
  echo "[run] ${dir}/${tag}"
  echo "  model = ${model}"
  echo "  out   = ${out_json}"
  echo "===================================================="
  export EVAL_OUT_JSON="${out_json}"
  export EVAL_MODEL_TAG="${RUN_KEY:-$dir}__${tag}"

  # train.py(mipha) 와 동일한 deepspeed 플래그.
  ( cd "${LLAVA_PHI}" && deepspeed --include "localhost:${CUDA_VISIBLE_DEVICES:-0}" --master_port "${MASTER_PORT:-29600}" "${WRAP}" \
      --deepspeed ./scripts/zero2.json \
      --model_name_or_path "${model}" \
      --version v0 \
      --data_path "${DUMMY_DATA}" \
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
