#!/bin/bash
# VILA-3B 임의 순서 커리큘럼 학습 (llava_curriculum/run_order_7B.sh 와 같은 방식).
#   bash run_order_3B.sh 2 0 4 3 1
# → checkpoint/VILA_3B_order_20431_nohex/{1_S2_Ishihara, 2_S0_LogMAR, ... , 5_S1_Similarity}
#   각 스테이지 출력의 end/ 가 다음 스테이지 입력으로 체이닝됨.
#   S1 은 기본 nohex 믹스처(s1_similarity_nohex). hex 로 돌리려면 S1_NOHEX=0
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <stage idx ...>   e.g. $0 2 0 4 3 1" >&2
  exit 1
fi

BASE_MODEL="/home/hail/vlm3/Model/VILA/VILA1.5-3b"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 단일 GPU 고정. 다른 GPU 쓰려면: CUDA_VISIBLE_DEVICES=1 bash run_order_3B.sh ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# run_all_3B.sh 와 동일한 학습 설정
export GRADIENT_CHECKPOINTING="False"
export DEEPSPEED_CONFIG="scripts/zero2.json"
export CONV_VERSION="vicuna_v1"
# 다른 VILA 학습과 동시에 돌 때 torchrun 포트 충돌 방지
export MASTER_PORT="${MASTER_PORT:-25051}"

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
export WANDB_PROJECT="vila_3b_curriculum_order_${TAG}"

OUT_ROOT="/home/hail/vlm3/checkpoint/VILA_3B_order_${TAG}"
mkdir -p "${OUT_ROOT}"
{ printf '%s\n' "${ORDER[@]/#/S}"; echo "s1_mixture=${S1_MIXTURE:-s1_similarity(hex)}"; } > "${OUT_ROOT}/order.txt"

cd /home/hail/vlm3/Model/VILA

echo "============================================"
echo "[VILA-3B Curriculum order=${ORDER[*]}] START"
echo "  base_model = ${BASE_MODEL}"
echo "  out_root   = ${OUT_ROOT}"
echo "  gpu        = ${CUDA_VISIBLE_DEVICES}"
echo "  s1_mixture = ${S1_MIXTURE:-s1_similarity(hex)}"
echo "============================================"

prev="${BASE_MODEL}"
pos=1
for i in "${ORDER[@]}"; do
  out_dir="${OUT_ROOT}/${pos}_${STAGE_NAME[$i]}"
  # wandb run id 가 스테이지/순서마다 유일하도록 (train.sh 가 RUN_NAME → WANDB_RUN_ID)
  RUN_NAME="${STAGE_NAME[$i]}_${TAG}" bash "${SCRIPT_DIR}/${STAGE_SCRIPT[$i]}" "${prev}" "${out_dir}"
  prev="${out_dir}/end"
  pos=$((pos + 1))
done

echo "============================================"
echo "[VILA-3B Curriculum order=${ORDER[*]}] DONE  → ${OUT_ROOT}"
echo "============================================"
