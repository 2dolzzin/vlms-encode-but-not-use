#!/bin/bash
# run_blank.sh
# counting_grid_blank 만 5개 모델(forward+reverse) 순차 실행.
# 모델별로 알맞은 conda env 를 activate → 실행 → deactivate 자동 처리.
# tmux 안에서 한 번만 실행하면 끝까지 돎.
#   bash run_blank.sh
#
# 한 모델이 실패해도 다음 모델로 계속 진행하고, 마지막에 성공/실패 요약을 찍는다.

cd "$(dirname "$0")"   # extract_benchmark/

TASK="counting_grid_blank"

# ── 모델별 (스크립트, conda env) 매핑 ───────────────────────────
#    ⚠️ mipha env 이름이 다르면 여기만 고치세요.
declare -A ENVS=(
  ["ben_llava_blind.py"]="llava"
  ["ben_mipha_blind.py"]="mipha"          # ← mipha env 명 확인
  ["ben_vila_3b_blind.py"]="vila"
  ["ben_vila_8b_blind.py"]="vila"
  ["ben_openflamingo_blind.py"]="openflamingo"
)
# 실행 순서 (빠른 것 → 느린 vila_8b 마지막)
ORDER=(ben_llava_blind.py ben_mipha_blind.py ben_vila_3b_blind.py ben_vila_8b_blind.py ben_openflamingo_blind.py)

# ── conda 훅 로드 (스크립트에서 conda activate 쓰려면 필요) ──────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
set +u                                   # conda 함수가 미정의 변수 참조 → nounset 이면 죽으므로 끔
source "${CONDA_BASE}/etc/profile.d/conda.sh"

declare -A STATUS
for script in "${ORDER[@]}"; do
  env="${ENVS[$script]}"
  echo ""
  echo "==================================================================="
  echo ">>> ${script}  (env: ${env})  task=${TASK}  $(date '+%F %T')"
  echo "==================================================================="
  conda deactivate 2>/dev/null || true
  conda activate "${env}" || { echo "[ERR] conda activate ${env} 실패"; STATUS[$script]="ENV_FAIL"; continue; }

  # 실패해도 스크립트 전체는 안 죽게
  if python "${script}" --run all --tasks "${TASK}"; then
    STATUS[$script]="OK"
  else
    STATUS[$script]="RUN_FAIL(exit $?)"
  fi
  conda deactivate 2>/dev/null || true
done

echo ""
echo "===================== 요약 ====================="
for script in "${ORDER[@]}"; do
  printf "  %-28s %s\n" "$script" "${STATUS[$script]:-SKIPPED}"
done
echo "완료: $(date '+%F %T')"
echo ""
echo "채점은 별도로 (아무 env 가능):"
echo "  for m in llava mipha vila_3b vila_8b openflamingo; do"
echo "    python acc_blind.py --model \${m}_blind         --task ${TASK}"
echo "    python acc_blind.py --model \${m}_reverse_blind --task ${TASK}"
echo "  done"
