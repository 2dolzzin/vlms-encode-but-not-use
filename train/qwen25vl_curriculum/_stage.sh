#!/bin/bash
# _stage.sh — 1 stage 공용 실행기. s0~s4 .sh 가 호출한다.
#   bash _stage.sh <STAGE_TAG> <IN_MODEL> <OUT_DIR> <DATA_PATH> <EVAL_DATA_PATH>
#
# 환경변수:
#   GPU_IDS    (기본 0,1) 사용할 물리 GPU. deepspeed --include localhost:<ids> 로 전달한다.
#              ⚠ deepspeed 런처는 CUDA_VISIBLE_DEVICES 를 자기 값으로 덮어쓰므로 --include 를 써야 한다.
#   NUM_GPUS   (기본: GPU_IDS 개수) 명시하면 GPU_IDS 와 개수가 맞아야 한다.
#   GLOBAL_BATCH (기본 16) grad_accum = GLOBAL_BATCH / NUM_GPUS. 나누어떨어져야 한다.
#              GPU 3장은 16 을 못 나누므로 GLOBAL_BATCH=15 (accum 5) 처럼 맞춰줘야 한다.
#              ⚠ 글로벌 배치를 바꾸면 총 step 수와 mid 지점도 바뀐다 → MID_STEPS 도 같이 조정.
#   MID_STEPS  (기본 1563) mid 체크포인트 step. 총 step 의 절반으로 잡는다.
#              (50000 샘플 기준: global 16 → 3125 step, mid 1563 / global 15 → 3334 step, mid 1667)
#   MASTER_PORT(기본 29700)
#   DS_CONFIG  (기본 zero2.json) scripts/ 안의 deepspeed 설정.
#              2 GPU 실측(S3=최장 시퀀스 기준): zero2 6.41s/it 피크 87GiB / zero3 9.09s/it 피크 63GiB.
#              PCIe 연결(NVLink 없음)이라 zero3 의 파라미터 all-gather 가 비싸서 zero2 가 40% 빠르다.
#              혹시 OOM 이 나면 DS_CONFIG=zero3.json 으로 바꾸면 된다 (메모리 24GiB 절약, 속도 40% 손해).
#   SEED       (기본 42), RUN_SUFFIX (wandb run 이름 접미사)
#   MAX_STEPS  (미설정) 설정하면 그 step 만 돌고 끝남 — 설정/경로 점검용 짧은 테스트에만 쓴다.
#   EVAL_AT_START/EVAL_AT_MID/EVAL_AT_END (기본 True)
#
# 출력: ${OUT_DIR}/checkpoint-1563 (mid, 모델만)  +  ${OUT_DIR}/end (final)  — llava/mipha 와 동일 레이아웃
set -euo pipefail

STAGE_TAG="$1"; IN_MODEL="$2"; OUT_DIR="$3"; DATA_PATH="$4"; EVAL_DATA_PATH="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-0,1}"
N_IDS=$(awk -F, '{print NF}' <<< "${GPU_IDS}")
NUM_GPUS="${NUM_GPUS:-${N_IDS}}"
if (( NUM_GPUS != N_IDS )); then
  echo "[ERR] NUM_GPUS=${NUM_GPUS} 와 GPU_IDS='${GPU_IDS}'(${N_IDS}개) 가 불일치"; exit 1
fi
MASTER_PORT="${MASTER_PORT:-29700}"
GLOBAL_BATCH="${GLOBAL_BATCH:-16}"
if (( GLOBAL_BATCH % NUM_GPUS != 0 )); then
  echo "[ERR] GLOBAL_BATCH=${GLOBAL_BATCH} 를 NUM_GPUS=${NUM_GPUS} 로 나눌 수 없음."
  echo "      GPU 3장이면 GLOBAL_BATCH=15 MID_STEPS=1667 처럼 맞춰서 실행하세요."; exit 1
fi
GRAD_ACCUM=$(( GLOBAL_BATCH / NUM_GPUS ))
MID_STEPS="${MID_STEPS:-1563}"

echo "===================================================="
echo "[${STAGE_TAG}]"
echo "  in_model  = ${IN_MODEL}"
echo "  out_dir   = ${OUT_DIR}"
echo "  data      = ${DATA_PATH}"
echo "  eval_data = ${EVAL_DATA_PATH}"
echo "  gpus      = ${GPU_IDS} (n=${NUM_GPUS}, grad_accum=${GRAD_ACCUM}, global=${GLOBAL_BATCH}, mid_step=${MID_STEPS})"
echo "===================================================="

mkdir -p "${OUT_DIR}/end"
export EVAL_DATA_PATH
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

deepspeed --include "localhost:${GPU_IDS}" --master_port "${MASTER_PORT}" \
    "${SCRIPT_DIR}/train_qwen.py" \
    --deepspeed "${SCRIPT_DIR}/scripts/${DS_CONFIG:-zero2.json}" \
    --model_name_or_path "${IN_MODEL}" \
    --data_path "${DATA_PATH}" \
    --image_folder "/" \
    --freeze_vision_tower True \
    --tune_merger True \
    --freeze_llm False \
    --bf16 True \
    --tf32 True \
    --output_dir "${OUT_DIR}/end" \
    --mid_save_root "${OUT_DIR}" \
    --mid_save_steps "${MID_STEPS}" \
    --eval_at_start "${EVAL_AT_START:-True}" \
    --eval_at_mid "${EVAL_AT_MID:-True}" \
    --eval_at_end "${EVAL_AT_END:-True}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-6 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --optim adamw_torch_fused \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to wandb \
    --seed "${SEED:-42}" \
    ${MAX_STEPS:+--max_steps ${MAX_STEPS}} \
    --run_name "${STAGE_TAG}${RUN_SUFFIX:-}"

echo "[${STAGE_TAG}] done → ${OUT_DIR}"
ls "${OUT_DIR}"
