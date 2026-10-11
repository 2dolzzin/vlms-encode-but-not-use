#!/bin/bash
# 임의 순서 커리큘럼 학습.
#   bash run_order_7B.sh 2 4 0 1 3
# → checkpoint/llava_order_24013/{1_S2_Ishihara, 2_S4_Depth, 3_S0_LogMAR, 4_S1_Similarity, 5_S3_Naming}
#   각 스테이지 출력의 end/ 가 다음 스테이지 입력으로 체이닝됨.
set -euo pipefail

export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800
# 단일 GPU 고정 (여러 장 보이면 nn.DataParallel로 빠져 dtype 에러 + 스텝수 변동).
# 기존 llava/ 체크포인트도 n_gpu=1 로 학습됨. 다른 GPU 쓰려면: CUDA_VISIBLE_DEVICES=1 bash run_order_7B.sh ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <stage idx ...>   e.g. $0 2 4 0 1 3" >&2
  exit 1
fi

BASE_MODEL="/home/hail/vlm3/Model/LLaVA/llava-v1.5-7b"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 스테이지 번호 → (스크립트, 출력 폴더명)
STAGE_SCRIPT=(s0_logmar.sh s1_similarity.sh s2_ishihara.sh s3_naming.sh s4_depth.sh)
STAGE_NAME=(S0_LogMAR S1_Similarity S2_Ishihara S3_Naming S4_Depth)

ORDER=("$@")
for i in "${ORDER[@]}"; do
  [[ "$i" =~ ^[0-4]$ ]] || { echo "invalid stage idx: $i (0~4)" >&2; exit 1; }
done

TAG="$(printf '%s' "${ORDER[@]}")"          # 2 4 0 1 3 → 24013

# S1 데이터: 기본 nohex(rebuttal 버전, forward/reverse S1_Similarity_nohex 와 동일). hex로 돌리려면 S1_NOHEX=0
S1_NOHEX="${S1_NOHEX:-1}"
if [ "${S1_NOHEX}" = "1" ]; then
  export S1_DATA="/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/rebuttal_nohex/similarity_non_flamingo_nohex.json"
  TAG="${TAG}_nohex"
else
  unset S1_DATA
fi
export WANDB_PROJECT="llava_curriculum_order_${TAG}"

# 시드: 기본 42 (HF Trainer 기본값 = 기존 llava/ 실험과 동일). 다르게 돌리면 폴더/런이름에 접미사.
#   SEED=123 bash run_order_7B.sh 2 4 0 1 3  → checkpoint/llava_order_24013_seed123/
export SEED="${SEED:-42}"
export RUN_SUFFIX=""
OUT_ROOT="/home/hail/vlm3/checkpoint/llava_order_${TAG}"
if [ "${SEED}" != "42" ]; then
  OUT_ROOT="${OUT_ROOT}_seed${SEED}"
  export RUN_SUFFIX="_seed${SEED}"
fi

mkdir -p "${OUT_ROOT}"
{ printf '%s\n' "${ORDER[@]/#/S}"; echo "seed=${SEED}"; echo "s1_data=${S1_DATA:-hex(default)}"; } > "${OUT_ROOT}/order.txt"   # 재현용 기록

cd /home/hail/vlm3/Model/LLaVA

echo "============================================"
echo "[Curriculum order=${ORDER[*]}] START"
echo "  base_model = ${BASE_MODEL}"
echo "  out_root   = ${OUT_ROOT}"
echo "  seed       = ${SEED}"
echo "  s1_data    = ${S1_DATA:-hex(default)}"
echo "============================================"

prev="${BASE_MODEL}"
pos=1
for i in "${ORDER[@]}"; do
  out_dir="${OUT_ROOT}/${pos}_${STAGE_NAME[$i]}"
  bash "${SCRIPT_DIR}/${STAGE_SCRIPT[$i]}" "${prev}" "${out_dir}"
  prev="${out_dir}/end"
  pos=$((pos + 1))
done

echo "============================================"
echo "[Curriculum order=${ORDER[*]}] DONE  → ${OUT_ROOT}"
echo "============================================"
