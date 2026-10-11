#!/usr/bin/env bash
# 6개 모델 S1_similarity 회귀 probing (inf, ridge) 일괄 실행.
# base python엔 wandb/sklearn 없으므로 llava env python으로 고정.
set -u
cd "$(dirname "$0")"

PY=/home/hail/anaconda3/envs/llava/bin/python
MODELS=(llava llava_reverse mipha mipha_reverse vila_3b vila_3b_reverse)

for m in "${MODELS[@]}"; do
  echo "================ MODEL $m  START $(date +%H:%M:%S) ================"
  if "$PY" probe_regressor.py --model "$m" --source inf --task S1_similarity --probe ridge; then
    echo "  DONE $m -> ./$m/inf_regression_{1,2,3}/ridge/"
  else
    echo "  FAILED $m (exit $?)"
  fi
done
echo "================ ALL DONE $(date +%H:%M:%S) ================"
