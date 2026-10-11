#!/bin/bash
# run_forgetting_openflamingo.sh
# ─────────────────────────────────────────────────────────────────────────────
# OpenFlamingo-3B(vitl-mpt1b) 커리큘럼 체크포인트들의 eval_loss(S0~S4) 측정 런처.
# eval_openflamingo.py(몽키패치)를 train.py(OF) 와 "동일한 deepspeed 플래그"로 런치.
# 차이: 스크립트=eval_openflamingo.py, output_dir=temp, save_strategy=no → 체크포인트 보호.
#
# 사용:
#   bash run_forgetting_openflamingo.sh            # fwd + rev 전부
#   DIRS=fwd bash run_forgetting_openflamingo.sh   # 정방향만
#   CUDA_VISIBLE_DEVICES=1 bash run_forgetting_openflamingo.sh
#
# 결과: results_openflamingo/{dir}__{tag}.json = {"model","model_path","losses":{S0..S4}}
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

OF_CUR="/home/hail/vlm3/train/openflamingo_curriculum"
OF_PKG="/home/hail/vlm3/Model/open_flamingo"
HERE="/home/hail/vlm3/train/eval_loss"
WRAP="${HERE}/eval_openflamingo.py"
RESULTS="${HERE}/results_openflamingo"
TMP_OUT="${HERE}/_tmp_out_openflamingo"
mkdir -p "${RESULTS}" "${TMP_OUT}"

export PYTHONPATH="${OF_PKG}:${OF_CUR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# eval 배치(속도 노브). eval_loss 는 고정 검증셋 평균이라 배치를 키워도 값 사실상 동일.
# 형제(mipha/vila/llava)와 bit-exact 비교가 필요하면 EVAL_BSZ=1 로.
EVAL_BSZ="${EVAL_BSZ:-16}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"

# make_supervised_data_module 가 학습셋을 만들 때 쓰는 더미(loss 무관, flamingo 포맷).
DUMMY_DATA="/home/hail/HDD/vlm_training/S0_Logmar/dataset/logmar_flamingo.json"

# ── 5개 task held-out eval 셋 (flamingo 포맷; S1 만 파일명 _v0) ──
export EVAL_SETS_JSON='[
  ["S0","/home/hail/HDD/vlm_validation/S0_Logmar/dataset/logmar_flamingo.json"],
  ["S1","/home/hail/HDD/vlm_validation/S1_Color_Similarity/dataset/color_similarity_flamingo_v0.json"],
  ["S2","/home/hail/HDD/vlm_validation/S2_Ishihara/dataset/ishihara_flamingo.json"],
  ["S3","/home/hail/HDD/vlm_validation/S3_Color_Naming/dataset/color_naming_flamingo.json"],
  ["S4","/home/hail/HDD/vlm_validation/S4_Depth/dataset/depth_flamingo.json"]
]'

# ── 평가할 체크포인트 시점들 ──
PRETRAINED="/home/hail/vlm3/Model/open_flamingo/ckpts/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"
FWD_ROOT="/home/hail/vlm3/checkpoint/OpenFlamingo"
REV_ROOT="/home/hail/vlm3/checkpoint/OpenFlamingo_reverse"
MID="${MID_STEP:-1562}"   # save_steps(train.sh) 와 동일. stage step<MID 면 mid 없음 → skip

# "tag|state"  (state = .pt 파일 또는 .pt 가 들어있는 디렉토리)
FWD_STATES=(
  "start|${PRETRAINED}"
  "S0_mid|${FWD_ROOT}/S0_LogMAR/checkpoint-${MID}"
  "S0_end|${FWD_ROOT}/S0_LogMAR"
  "S1_mid|${FWD_ROOT}/S1_Similarity/checkpoint-${MID}"
  "S1_end|${FWD_ROOT}/S1_Similarity"
  "S2_mid|${FWD_ROOT}/S2_Ishihara/checkpoint-${MID}"
  "S2_end|${FWD_ROOT}/S2_Ishihara"
  "S3_mid|${FWD_ROOT}/S3_Naming/checkpoint-${MID}"
  "S3_end|${FWD_ROOT}/S3_Naming"
  "S4_mid|${FWD_ROOT}/S4_Depth/checkpoint-${MID}"
  "S4_end|${FWD_ROOT}/S4_Depth"
)
REV_STATES=(
  "start|${PRETRAINED}"
  "S4_mid|${REV_ROOT}/S4_Depth/checkpoint-${MID}"
  "S4_end|${REV_ROOT}/S4_Depth"
  "S3_mid|${REV_ROOT}/S3_Naming/checkpoint-${MID}"
  "S3_end|${REV_ROOT}/S3_Naming"
  "S2_mid|${REV_ROOT}/S2_Ishihara/checkpoint-${MID}"
  "S2_end|${REV_ROOT}/S2_Ishihara"
  "S1_mid|${REV_ROOT}/S1_Similarity/checkpoint-${MID}"
  "S1_end|${REV_ROOT}/S1_Similarity"
  "S0_mid|${REV_ROOT}/S0_LogMAR/checkpoint-${MID}"
  "S0_end|${REV_ROOT}/S0_LogMAR"
)

# state(파일 또는 디렉토리) → 실제 .pt 파일 경로로 해석. 없으면 빈 문자열.
resolve_ckpt () {
  local p="$1"
  if [[ -f "${p}" ]]; then echo "${p}"; return 0; fi
  if [[ -d "${p}" ]]; then
    for c in flamingo_model.pt pytorch_model.bin; do
      [[ -f "${p}/${c}" ]] && { echo "${p}/${c}"; return 0; }
    done
  fi
  echo ""
}

run_one () {
  local dir="$1" tag="$2" state="$3"
  local out_json="${RESULTS}/${RUN_KEY:-$dir}__${tag}.json"
  if [[ -f "${out_json}" ]]; then
    echo "[skip] ${dir}/${tag} (이미 있음)"; return 0
  fi
  local ckpt; ckpt="$(resolve_ckpt "${state}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[WARN] 체크포인트 없음, 건너뜀: ${state}"; return 0
  fi
  echo "===================================================="
  echo "[run] ${dir}/${tag}"
  echo "  ckpt = ${ckpt}"
  echo "  out  = ${out_json}"
  echo "===================================================="
  export EVAL_OUT_JSON="${out_json}"
  export EVAL_MODEL_TAG="${RUN_KEY:-$dir}__${tag}"

  # train.py(OF) 와 동일한 deepspeed 플래그. (학습은 0스텝, save 안 함)
  deepspeed --include "localhost:${CUDA_VISIBLE_DEVICES:-0}" \
      --master_port "${MASTER_PORT:-29611}" "${WRAP}" \
      --deepspeed "${OF_CUR}/scripts/zero2.json" \
      --vision_encoder_path "ViT-L-14" \
      --vision_encoder_pretrained "openai" \
      --lm_path "anas-awadalla/mpt-1b-redpajama-200b" \
      --tokenizer_path "anas-awadalla/mpt-1b-redpajama-200b" \
      --cross_attn_every_n_layers 1 \
      --train_mode "adapter" \
      --freeze_vision_encoder True \
      --freeze_lm_embeddings True \
      --pretrained_checkpoint "${ckpt}" \
      --data_path "${DUMMY_DATA}" \
      --image_folder "/" \
      --image_aspect_ratio "pad" \
      --lazy_preprocess True \
      --bf16 True \
      --tf32 True \
      --output_dir "${TMP_OUT}" \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size "${EVAL_BSZ}" \
      --gradient_accumulation_steps 16 \
      --evaluation_strategy "no" \
      --save_strategy "no" \
      --learning_rate 1e-6 \
      --weight_decay 0.0 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 10 \
      --model_max_length 2048 \
      --gradient_checkpointing False \
      --dataloader_num_workers "${EVAL_WORKERS}" \
      --report_to none \
      --run_name "forget_${dir}_${tag}"
  local rc=$?
  if [[ ! -f "${out_json}" ]]; then
    echo "[ERROR] 결과 JSON 미생성 (${dir}/${tag}), rc=${rc}"
  fi
}


# ── 무작위 순서 런 (RUN_KEY=<model>_order_<tag>, ORDER_ROOT=체크포인트 루트): 상태 목록을 폴더에서 자동 구성, S1 검증셋은 nohex ──
if [[ -n "${RUN_KEY:-}" ]]; then
  : "${ORDER_ROOT:?RUN_KEY 사용 시 ORDER_ROOT 필요}"
  mapfile -t ORDER_STATES < <(python3 /home/hail/vlm3/order_runs.py states "${ORDER_ROOT}" --base "${PRETRAINED}" --mid "checkpoint-${MID}" --end ".")
  export EVAL_SETS_JSON="$(S1_NOHEX_VAL="/home/hail/HDD/vlm_validation/S1_Color_Similarity/dataset/rebuttal_nohex/color_similarity_flamingo_nohex.json" python3 -c 'import json,os; s=json.loads(os.environ["EVAL_SETS_JSON"]); print(json.dumps([[k, os.environ["S1_NOHEX_VAL"] if k=="S1" else p] for k,p in s]))')"
  DIRS="order"
  echo "[order] ${RUN_KEY}: ${#ORDER_STATES[@]} 상태, root=${ORDER_ROOT}"
fi
DIRS="${DIRS:-fwd rev}"
for d in ${DIRS}; do
  if [[ "$d" == "fwd" ]]; then states=("${FWD_STATES[@]}"); elif [[ "$d" == "order" ]]; then states=("${ORDER_STATES[@]}"); else states=("${REV_STATES[@]}"); fi
  for line in "${states[@]}"; do
    tag="${line%%|*}"; state="${line#*|}"
    run_one "$d" "$tag" "$state"
  done
done

echo "[done] 결과 → ${RESULTS}/"
