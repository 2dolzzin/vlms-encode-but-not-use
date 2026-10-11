#!/usr/bin/env bash
# 모든 figure 재생성 → figure/exp{1..4}_*/ (입력: csv_out/*.csv, 먼저 python tools/export_csv.py 실행 권장)
cd "$(dirname "$0")"
for s in curriculum_inf curriculum_inf5 curriculum_probing curriculum_probing_permodel \
         benchmark_inf benchmark_inf5 benchmark_probing benchmark_probing_permodel \
         order_inf order_probing_permodel; do
  echo "== $s"; python3 "$s.py" || exit 1
done
