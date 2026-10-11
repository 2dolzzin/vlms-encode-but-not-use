#!/bin/bash
# run_forgetting_vila.sh
# ─────────────────────────────────────────────────────────────────────────────
# VILA(3B/8B) 커리큘럼 체크포인트들의 eval_loss(S0~S4) 측정 런처.
# eval_vila.py(몽키패치)를 train_mem 과 "동일한 torchrun+deepspeed 플래그"로 런치.
# 차이: 스크립트=eval_vila.py, output_dir=temp, save_strategy=no → 체크포인트 보호.
#
# 사용:
#   bash run_forgetting_vila.sh            # 3B, fwd+rev
#   bash run_forgetting_vila.sh 3b         # 동일
#   bash run_forgetting_vila.sh 8b         # 8B (fwd+rev)
#   DIRS=fwd bash run_forgetting_vila.sh 3b
#
# 결과: results_vila_{3b|8b}/{dir}__{tag}.json = {"model","model_path","losses":{S0..S4}}
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

VILA_DIR="/home/hail/vlm3/Model/VILA"
HERE="/home/hail/vlm3/train/eval_loss"
WRAP="${HERE}/eval_vila.py"

MODEL="${1:-3b}"   # 3b | 8b

# ── 모델별 설정 (run_all_3B.sh / run_all_8B.sh 와 동일) ──
if [[ "$MODEL" == "3b" ]]; then
  BASE_MODEL="/home/hail/vlm3/Model/VILA/VILA1.5-3b"
  FWD_ROOT="/home/hail/vlm3/checkpoint/VILA_3B"
  REV_ROOT="/home/hail/vlm3/checkpoint/VILA_3B_reverse"
  CONV_VERSION="vicuna_v1"
  DEEPSPEED_CONFIG="scripts/zero2.json"
  GRADIENT_CHECKPOINTING="False"
  DEFAULT_DIRS="fwd rev"
elif [[ "$MODEL" == "8b" ]]; then
  BASE_MODEL="/home/hail/vlm3/Model/VILA/Llama-3-VILA1.5-8B"
  FWD_ROOT="/home/hail/vlm3/checkpoint/VILA_8B"
  REV_ROOT="/home/hail/vlm3/checkpoint/VILA_8B_reverse"   # reverse 학습 완료(디스크에 존재)
  CONV_VERSION="llama_3"
  # eval 전용: param 은 GPU 상주(빠른 forward), optimizer 만 CPU offload(미사용·OOM 방지).
  # gradient_checkpointing 은 eval(backward 없음)에선 재계산 오버헤드뿐이라 끔. loss 값 불변.
  DEEPSPEED_CONFIG="scripts/zero3_evalfast.json"
  GRADIENT_CHECKPOINTING="False"
  DEFAULT_DIRS="fwd rev"
else
  echo "usage: bash $0 [3b|8b]" >&2; exit 1
fi

RESULTS="${HERE}/results_vila_${MODEL}"
TMP_OUT="${HERE}/_tmp_out_vila_${MODEL}"
mkdir -p "${RESULTS}" "${TMP_OUT}"

# 더미 학습셋(레지스트리 이름). loss 무관, train_dataset 빌드만 통과시키려는 용도.
DUMMY_MIXTURE="s0_logmar"

export EVAL_SETS_JSON='[
  ["S0","/home/hail/HDD/vlm_validation/S0_Logmar/dataset/logmar_non_flamingo.json"],
  ["S1","/home/hail/HDD/vlm_validation/S1_Color_Similarity/dataset/color_similarity_non_flamingo.json"],
  ["S2","/home/hail/HDD/vlm_validation/S2_Ishihara/dataset/ishihara_non_flamingo.json"],
  ["S3","/home/hail/HDD/vlm_validation/S3_Color_Naming/dataset/color_naming_non_flamingo.json"],
  ["S4","/home/hail/HDD/vlm_validation/S4_Depth/dataset/depth_non_flamingo.json"]
]'

# "tag|model_path"  (start=base, stage 별 mid=checkpoint-1563 / end). dir 은 -phi2 없음.
build_states () {   # $1=root  → 전역 STATES 채움 (정방향 S0→S4 순)
  local root="$1"
  STATES=(
    "start|${BASE_MODEL}"
    "S0_mid|${root}/S0_LogMAR/checkpoint-1563"   "S0_end|${root}/S0_LogMAR/end"
    "S1_mid|${root}/S1_Similarity/checkpoint-1563" "S1_end|${root}/S1_Similarity/end"
    "S2_mid|${root}/S2_Ishihara/checkpoint-1563"   "S2_end|${root}/S2_Ishihara/end"
    "S3_mid|${root}/S3_Naming/checkpoint-1563"     "S3_end|${root}/S3_Naming/end"
    "S4_mid|${root}/S4_Depth/checkpoint-1563"      "S4_end|${root}/S4_Depth/end"
  )
}

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
  echo "[run] ${MODEL} ${dir}/${tag}"
  echo "  model = ${model}"
  echo "  out   = ${out_json}"
  echo "===================================================="
  export EVAL_OUT_JSON="${out_json}"
  export EVAL_MODEL_TAG="${RUN_KEY:-$dir}__${tag}"

  # VILA 통신 최적화 env (train.sh 와 동일)
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  export DS_SKIP_CUDA_CHECK=1
  export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
  export PYTORCH_ALLOC_CONF="expandable_segments:True"
  export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=1

  # train_mem 과 동일한 torchrun+deepspeed. 차이: WRAP, output_dir=temp, save no.
  ( cd "${VILA_DIR}" && torchrun \
      --nnodes=1 --nproc_per_node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-25011}" \
      "${WRAP}" \
        --deepspeed "${DEEPSPEED_CONFIG}" \
        --model_name_or_path "${model}" \
        --version "${CONV_VERSION}" \
        --data_mixture "${DUMMY_MIXTURE}" \
        --image_folder "/" \
        --vision_tower "${model}/vision_tower" \
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
        --output_dir "${TMP_OUT}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy no \
        --save_strategy no \
        --learning_rate 1e-6 \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --model_max_length 2048 \
        --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
        --dataloader_num_workers 8 \
        --lazy_preprocess True \
        --report_to none )
  local rc=$?
  if [[ ! -f "${out_json}" ]]; then
    echo "[ERROR] 결과 JSON 미생성 (${MODEL} ${dir}/${tag}), rc=${rc}"
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
DIRS="${DIRS:-$DEFAULT_DIRS}"
for d in ${DIRS}; do
  if [[ "$d" == "fwd" ]]; then
    build_states "${FWD_ROOT}"
  elif [[ "$d" == "order" ]]; then
    STATES=("${ORDER_STATES[@]}")
  else
    if [[ -z "${REV_ROOT}" ]]; then
      echo "[skip] ${MODEL} rev (reverse 체크포인트 없음)"; continue
    fi
    build_states "${REV_ROOT}"
  fi
  for line in "${STATES[@]}"; do
    tag="${line%%|*}"; model="${line#*|}"
    run_one "$d" "$tag" "$model"
  done
done

echo "[done] 결과 → ${RESULTS}/"
