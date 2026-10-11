# vlm3 — VLM 발달 커리큘럼 학습 실험

5개 시각 과제(S0 LogMAR → S1 Color Similarity → S2 Ishihara → S3 Color Naming → S4 Depth)를
순서대로 fine-tuning 하면서, 과제 수행·망각·내부 표현(프로빙)·외부 벤치마크 전이가
학습 순서(forward / reverse / 무작위 3종)에 따라 어떻게 달라지는지 본다.

- 모델: LLaVA-1.5-7B, Mipha-3B, VILA-3B, OpenFlamingo-3B, Qwen2.5-VL-7B (VILA-8B 는 fwd/rev 만, 그림에서는 제외)
- 런 키: `<model>`, `<model>_reverse`, `<model>_order_<tag>` (tag = 14203 / 20431 / 31024, S-index 순서)
- 시점: Base + 5단계 × {mid(checkpoint-1563), end} = 11

## 디렉토리

```
order_runs.py          런 키 ↔ 체크포인트 폴더 매핑 공용 모듈 (여러 스크립트가 import)
run_eval.sh            런 하나를 끝까지 평가: 5-task 추론 → blind 추론 → 채점 2종
tools/                 집계: export_csv.py(variant별 CSV), make_experiment_csvs.py(실험별 CSV),
                       make_tables.py(모델별 md 표), summarize_results.py, status.py(진행 현황)
train/                 커리큘럼 학습 스크립트 (모델별 폴더, run_all*.sh / run_order*.sh),
                       eval_loss/ (망각 측정: 11시점 × 5검증셋 eval loss)
extract_5_tasks/       Exp1 5-task 추론 (inf_<model>.py, acc_parsing.py) + s1_rebuttal/ (리버탈용 S1 변형 자극)
extract_benchmark/     Exp3 BlindTest 8 + line_plot + MMVP 추론 (ben_<model>_blind.py, acc_blind.py)
probing/               Exp2/4 프로빙 (probe_classifier.py MLP|linear, probe_regressor.py ridge, run_probing.sh)
                       rebuttal_exp/ (리버탈 보조 실험: 선형 battery, S4 token-level)
figure/                그림 스크립트 + exp1~4 디렉토리별 출력 (figure/README.md)
csv_out/               집계 CSV (variant별) + experiments/ (실험별, 행=모델·열=11시점) + tables/
glmm/                  R GLMM 분석
ACCV_2026_rebuttal_template/
checkpoint/  Model/    git 제외 (아래)
```

## 파이프라인

```bash
# 1) 학습  (모델별 폴더, conda env 는 스크립트 안에서 활성화)
bash train/llava_curriculum/run_order_7B.sh 2 0 4 3 1        # 무작위 순서 예시 → checkpoint/llava_order_20431_nohex/
# 2) 평가 (5-task + blind + 채점)
CUDA_VISIBLE_DEVICES=0 bash run_eval.sh llava_order_20431
# 3) 프로빙 (MLP + ridge; --with-linear 로 GPU 선형 프로브 추가)
CUDA_VISIBLE_DEVICES=0 bash probing/run_probing.sh --with-linear llava_order_20431
# 4) 망각(eval loss)
bash train/eval_loss/run_forgetting.sh llava_order_20431
# 5) 집계 → 표 → 그림
python tools/export_csv.py && python tools/make_experiment_csvs.py --no-refresh && python tools/make_tables.py
cd figure && ./make_all_figures.sh
python tools/status.py        # 전체 진행 현황
```

## git 에 포함되지 않는 것 (서버 `/home/hail/vlm3` 에만 있음)

| 경로 | 내용 | 복구 방법 |
|---|---|---|
| `checkpoint/` | 학습 체크포인트 (~3T) | train/ 스크립트로 재학습 |
| `Model/` | 서드파티 코드·가중치 | 아래 표대로 clone / 다운로드 |
| `*/result*/` 의 embeddings, raw_responses, parsing | 추론 원시 출력 | run_eval.sh |
| `probing/<run>/` | 프로빙 원시 json (3시드) | run_probing.sh (집계본은 csv_out/) |
| `eval_logs/`, `*.log`, `wandb/` | 로그 | — |
| `figure/**/*.png` | 그림 png (pdf 는 추적) | figure/make_all_figures.sh |

`Model/` 구성 (2026-10 기준 커밋; 일부 로컬 수정 있음 → 서버 사본 참고):

| 폴더 | 출처 |
|---|---|
| Model/LLaVA | https://github.com/haotian-liu/LLaVA @ c121f04 |
| Model/llava-phi (Mipha) | https://github.com/xmoanvaf/llava-phi @ 21dafbf |
| Model/open_flamingo | https://github.com/mlfoundations/open_flamingo @ 655f693 |
| Model/VILA | https://github.com/Efficient-Large-Model/VILA @ 6b941da |
| Model/Qwen2.5-VL-7B-Instruct | HF `Qwen/Qwen2.5-VL-7B-Instruct` 가중치 |

## 결과 읽을 때 주의

- S1 Color Similarity 는 MAE(낮을수록 좋음). 표·그림에서는 1−MAE 로 표시.
  추론용 78쌍이 학습셋에 포함되어 S1 학습 직후 1.000 은 암기이며, fwd/rev(hex 프롬프트)와 random/Qwen(nohex)은 직접 비교 불가.
- MMVP 추론은 pair_accuracy(쌍 둘 다 정답, chance 1/4). 프로빙은 146-class 분류.
- 파싱 전멸(형식 붕괴)은 0 이 아니라 빈칸/NaN.
- 프로빙 그림은 MLP 가 기본, `_linear` 접미사 = GPU 선형 분류기 (S0 LogMAR 얕은 층에서만 두 분류기 차이가 큼).
