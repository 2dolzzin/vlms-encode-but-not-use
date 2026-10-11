#!/usr/bin/env bash
# run_qwen25vl_probing.sh — Qwen2.5-VL-7B (fwd/rev) 임베딩으로 기존 probing 배터리 일괄 실행.
# probing 스크립트들은 --model 만 바꾸면 되므로 새 코드 없이 호출만 한다.
#   conda activate qwenvl   (sklearn/scipy/wandb 포함)
#   bash run_qwen25vl_probing.sh            # inf + ben 전부
#   SOURCES="inf" bash run_qwen25vl_probing.sh
# 결과: probing/qwen25vl_7b{,_reverse}/{inf,ben}_{classification,regression,classification_logreg}_{1,2,3}/
set -u
cd "$(dirname "$0")"

MODELS=(${MODELS:-qwen25vl_7b qwen25vl_7b_reverse})
SOURCES=(${SOURCES:-inf ben})
JOBS="${JOBS:-16}"

for m in "${MODELS[@]}"; do
  for s in "${SOURCES[@]}"; do
    echo "================ ${m} / ${s}  $(date +%H:%M:%S) ================"
    # 이산 과제: MLP probe (본 분석) + 선형 logreg 대조군
    python probe_classifier.py --model "$m" --source "$s"             || echo "  [FAIL] classifier ${m}/${s}"
    python probe_logreg.py     --model "$m" --source "$s" --jobs "$JOBS" || echo "  [FAIL] logreg ${m}/${s}"
    # 연속 과제: inf=S1_similarity, ben=counting_grid (ridge 가 본 분석)
    if [[ "$s" == "inf" ]]; then
      python probe_regressor.py --model "$m" --source inf --task S1_similarity --probe ridge || echo "  [FAIL] regressor ${m}/inf"
    else
      python probe_regressor.py --model "$m" --source ben --task counting_grid --probe ridge || echo "  [FAIL] regressor ${m}/ben"
    fi
  done
done
echo "================ ALL DONE $(date +%H:%M:%S) ================"
