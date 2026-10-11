#!/bin/bash
# setup_env.sh — Qwen2.5-VL 전용 conda env(qwenvl) 생성 + 베이스 모델 다운로드.
#   bash setup_env.sh            # env 생성 + 모델 다운로드
#   bash setup_env.sh --no-model # env 만
#
# 왜 별도 env 인가: llava(4.37)/mipha,openflamingo(4.38)/vila(4.46) env 의 transformers 는
# Qwen2.5-VL(>=4.49) 을 못 읽고, base env 에는 deepspeed 가 없다. 기존 env 는 건드리지 않는다.
set -euo pipefail

ENV_NAME="qwenvl"
MODEL_DIR="/home/hail/vlm3/Model/Qwen2.5-VL-7B-Instruct"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] env ${ENV_NAME} 이미 있음 → 패키지 설치만 재확인"
else
  conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"

# GPU: RTX PRO 6000 Blackwell(sm_120) → CUDA 12.8+ 빌드 필요. 시스템 nvcc 가 13.0 이라 cu130 사용.
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "transformers==4.57.6" "accelerate>=1.10" "deepspeed>=0.17" \
            "qwen-vl-utils>=0.0.11" wandb scikit-learn scipy tqdm pillow \
            sentencepiece protobuf "huggingface_hub[cli]" safetensors

python - <<'PY'
import torch, transformers, deepspeed, accelerate
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("transformers", transformers.__version__, "deepspeed", deepspeed.__version__, "accelerate", accelerate.__version__)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # noqa
print("Qwen2.5-VL import OK")
PY

if [[ "${1:-}" != "--no-model" ]]; then
  if [[ -f "${MODEL_DIR}/config.json" ]]; then
    echo "[setup] 모델 이미 있음: ${MODEL_DIR}"
  else
    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "${MODEL_DIR}"
  fi
fi
echo "[setup] DONE  (env=${ENV_NAME}, model=${MODEL_DIR})"
