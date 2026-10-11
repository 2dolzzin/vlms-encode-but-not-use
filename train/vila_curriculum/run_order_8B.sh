#!/bin/bash
# VILA-8B 임의 순서 커리큘럼 학습 (run_order_3B.sh 의 8B 판, 설정은 run_all_8B.sh 와 동일).
#   CUDA_VISIBLE_DEVICES=0 bash run_order_8B.sh 2 0 4 3 1
# → checkpoint/VILA_8B_order_20431_nohex/{1_S2_Ishihara, 2_S0_LogMAR, ... , 5_S1_Similarity}
#   각 스테이지 출력의 end/ 가 다음 스테이지 입력으로 체이닝됨.
#   S1 은 기본 nohex 믹스처(s1_similarity_nohex). hex 로 돌리려면 S1_NOHEX=0
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: CUDA_VISIBLE_DEVICES=<id> $0 <stage idx ...>   e.g. $0 2 0 4 3 1" >&2
  exit 1
fi

BASE_MODEL="/home/hail/vlm3/Model/VILA/Llama-3-VILA1.5-8B"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 단일 GPU 고정(zero3_offload). 다른 GPU: CUDA_VISIBLE_DEVICES=1 bash run_order_8B.sh ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# run_all_8B.sh 와 동일한 학습 설정
export GLOBAL_TRAIN_BATCH_SIZE=16
export GRADIENT_ACCUMULATION_STEPS=4
export GRADIENT_CHECKPOINTING="True"
export DEEPSPEED_CONFIG="scripts/zero3_offload.json"
export CONV_VERSION="llama_3"
# 다른 VILA 학습과 동시에 돌 때 torchrun 포트 충돌 방지
export MASTER_PORT="${MASTER_PORT:-25061}"

# 스테이지 번호 → (스크립트, 출력 폴더명)
STAGE_SCRIPT=(s0_logmar.sh s1_similarity.sh s2_ishihara.sh s3_naming.sh s4_depth.sh)
STAGE_NAME=(S0_LogMAR S1_Similarity S2_Ishihara S3_Naming S4_Depth)

ORDER=("$@")
for i in "${ORDER[@]}"; do
  [[ "$i" =~ ^[0-4]$ ]] || { echo "invalid stage idx: $i (0~4)" >&2; exit 1; }
done

TAG="$(printf '%s' "${ORDER[@]}")"          # 2 0 4 3 1 → 20431

# S1 믹스처: 기본 nohex (S1_Similarity_nohex 와 동일한 데이터)
S1_NOHEX="${S1_NOHEX:-1}"
if [ "${S1_NOHEX}" = "1" ]; then
  export S1_MIXTURE="s1_similarity_nohex"
  TAG="${TAG}_nohex"
else
  unset S1_MIXTURE
fi
export WANDB_PROJECT="vila_8b_curriculum_order_${TAG}"

OUT_ROOT="/home/hail/vlm3/checkpoint/VILA_8B_order_${TAG}"
mkdir -p "${OUT_ROOT}"
{ printf '%s\n' "${ORDER[@]/#/S}"; echo "s1_mixture=${S1_MIXTURE:-s1_similarity(hex)}"; } > "${OUT_ROOT}/order.txt"

cd /home/hail/vlm3/Model/VILA

echo "============================================"
echo "[VILA-8B Curriculum order=${ORDER[*]}] START"
echo "  base_model = ${BASE_MODEL}"
echo "  out_root   = ${OUT_ROOT}"
echo "  gpu        = ${CUDA_VISIBLE_DEVICES}"
echo "  s1_mixture = ${S1_MIXTURE:-s1_similarity(hex)}"
echo "============================================"

# START_POS=N 이면 N 번째 스테이지부터 재개한다 (N-1 번째의 end/ 를 입력으로).
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
  # wandb run id 가 스테이지/순서마다 유일하도록 (train.sh 가 RUN_NAME → WANDB_RUN_ID)
  RUN_NAME="${STAGE_NAME[$i]}_${TAG}" bash "${SCRIPT_DIR}/${STAGE_SCRIPT[$i]}" "${prev}" "${out_dir}"
  # 디스크 절약: mid 체크포인트의 DeepSpeed 옵티마이저 상태(global_step*, 8B 는 스테이지당 ~91G)는 평가에 안 쓰이고
  # 다음 스테이지는 end/ 에서 이어가므로 스테이지가 끝나면 바로 지운다. 보존하려면 KEEP_OPTIM=1.
  if [ "${KEEP_OPTIM:-0}" != "1" ]; then
    for g in "${out_dir}"/checkpoint-*/global_step*; do
      [ -d "$g" ] || continue
      echo "[cleanup] rm -rf $g ($(du -sh "$g" 2>/dev/null | cut -f1))"; rm -rf "$g"
    done
  fi
  prev="${out_dir}/end"
  pos=$((pos + 1))
done

echo "============================================"
echo "[VILA-8B Curriculum order=${ORDER[*]}] DONE  → ${OUT_ROOT}"
echo "============================================"
