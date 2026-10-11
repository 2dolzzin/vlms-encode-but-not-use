#!/bin/bash
# Qwen2.5-VL-7B 임의 순서 커리큘럼 학습 (llava_curriculum/run_order_7B.sh 와 같은 방식).
#   GPU_IDS=0,1 bash run_order.sh 1 4 2 0 3
# → checkpoint/Qwen25VL_7B_order_14203_nohex/{1_S1_Similarity, 2_S4_Depth, ...}
#   S1 은 스테이지 스크립트 기본값이 이미 nohex.
set -euo pipefail
export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800

if [ "$#" -lt 1 ]; then
  echo "usage: GPU_IDS=<ids> $0 <stage idx ...>   e.g. GPU_IDS=0,1 $0 1 4 2 0 3" >&2
  exit 1
fi

BASE_MODEL="/home/hail/vlm3/Model/Qwen2.5-VL-7B-Instruct"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# deepspeed 런처가 CUDA_VISIBLE_DEVICES 를 덮어쓰므로 GPU_IDS(--include)로 지정한다.
export GPU_IDS="${GPU_IDS:-0,1}"
export MASTER_PORT="${MASTER_PORT:-29700}"

STAGE_SCRIPT=(s0_logmar.sh s1_similarity.sh s2_ishihara.sh s3_naming.sh s4_depth.sh)
STAGE_NAME=(S0_LogMAR S1_Similarity S2_Ishihara S3_Naming S4_Depth)

ORDER=("$@")
for i in "${ORDER[@]}"; do
  [[ "$i" =~ ^[0-4]$ ]] || { echo "invalid stage idx: $i (0~4)" >&2; exit 1; }
done

TAG="$(printf '%s' "${ORDER[@]}")_nohex"
export RUN_SUFFIX="_${TAG}"
export WANDB_PROJECT="qwen25vl_curriculum_order_${TAG}"
OUT_ROOT="/home/hail/vlm3/checkpoint/Qwen25VL_7B_order_${TAG}"

mkdir -p "${OUT_ROOT}"
printf '%s\n' "${ORDER[@]/#/S}" > "${OUT_ROOT}/order.txt"

echo "============================================"
echo "[Qwen2.5-VL Curriculum order=${ORDER[*]}] START"
echo "  base_model = ${BASE_MODEL}"
echo "  out_root   = ${OUT_ROOT}"
echo "  gpu_ids    = ${GPU_IDS}  (master_port=${MASTER_PORT})"
echo "============================================"

# START_POS=N 이면 N 번째 스테이지부터 재개한다 (N-1 번째의 end/ 를 입력으로).
#   예) 2단계에서 끊긴 경우: START_POS=2 bash run_order.sh 3 1 0 2 4
START_POS="${START_POS:-1}"
prev="${BASE_MODEL}"
if (( START_POS > 1 )); then
  pi=$(( START_POS - 2 ))
  prev="${OUT_ROOT}/$(( START_POS - 1 ))_${STAGE_NAME[${ORDER[$pi]}]}/end"
  [ -d "${prev}" ] || { echo "[ERR] 재개 입력이 없음: ${prev}" >&2; exit 1; }
  echo "  resume     : ${START_POS} 단계부터, 입력 = ${prev}"
fi

pos=1
for i in "${ORDER[@]}"; do
  out_dir="${OUT_ROOT}/${pos}_${STAGE_NAME[$i]}"
  if (( pos < START_POS )); then
    echo "[skip] ${pos}_${STAGE_NAME[$i]} (이미 완료)"
    pos=$((pos + 1)); continue
  fi
  bash "${SCRIPT_DIR}/${STAGE_SCRIPT[$i]}" "${prev}" "${out_dir}"
  prev="${out_dir}/end"
  pos=$((pos + 1))
done

echo "============================================"
echo "[Qwen2.5-VL Curriculum order=${ORDER[*]}] DONE  → ${OUT_ROOT}"
echo "============================================"
