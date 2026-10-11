# figure/

`./make_all_figures.sh` 로 전부 재생성 (입력 = `csv_out/*.csv`, 먼저 `python tools/export_csv.py`).
출력은 실험별 디렉토리. 각 그림은 png + pdf.

| 디렉토리 | 실험 | 파일 | 내용 |
|---|---|---|---|
| exp1_curriculum_inference | Exp1 5-task 추론 | curriculum_inf, curriculum_inf5 | fwd/rev, 선=모델 (3열 / 5열 레이아웃) |
| | | curriculum_inf_orders | 블록=순서 5개(fwd/rev/랜덤3), 선=모델, x=그 순서의 학습 진행 |
| | | curriculum_inf_bystage | x=스테이지 이름 고정, 선=순서 5개, 패널=task×모델 |
| exp2_curriculum_probing | Exp2 5-task 프로빙 | curriculum_probing | fwd/rev 전 모델 합본 (MLP 분류기) |
| | | curriculum_probing_<model> | 모델별 fwd/rev |
| | | curriculum_probing_orders_<model> | 모델별 순서 5블록 |
| exp3_benchmark_inference | Exp3 blind 8 + line_plot + MMVP 추론 | benchmark_inf, benchmark_inf5, benchmark_inf_orders, benchmark_inf_bystage | exp1 과 같은 구성 |
| exp4_benchmark_probing | Exp4 벤치마크 프로빙 | benchmark_probing_1 (LLaVA/Mipha/VILA-3B), benchmark_probing_2 (OF/Qwen) | fwd/rev 합본 2분할 |
| | | benchmark_probing_<model>, benchmark_probing_orders_<model> | exp2 와 같은 구성 |

- 모델: LLaVA-1.5, Mipha-3B, VILA-3B, OpenFlamingo, Qwen2.5-VL (VILA-8B 제외).
- MMVP 추론은 pair_accuracy(chance 1/4). S1 Color Sim 은 1−MAE 로 표시.
- 프로빙 그림은 **MLP 분류기**가 기본, 같은 이름에 `_linear` 가 붙은 파일은 **GPU 선형 분류기** 버전
  (예: `curriculum_probing_linear_llava`, `benchmark_probing_orders_linear_mipha`). S1 ridge 회귀(1−MAE)는 두 버전 공통.
