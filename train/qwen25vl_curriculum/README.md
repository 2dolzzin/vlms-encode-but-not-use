# Qwen2.5-VL-7B 커리큘럼 파이프라인

llava / mipha / vila / openflamingo 와 동일한 실험(커리큘럼 학습 → 11시점 ckpt → 5-task 추론+임베딩 → 벤치마크/blind → 프로빙 → 망각 eval_loss)을 Qwen2.5-VL-7B-Instruct 에 대해 돌리기 위한 코드 모음.

| 단계 | 파일 | 결과 위치 |
|---|---|---|
| env / 모델 | `setup_env.sh` (conda env `qwenvl`, `Model/Qwen2.5-VL-7B-Instruct`) | |
| 학습 | `run_all.sh` / `run_all_reverse.sh` → `s0~s4_*.sh` → `_stage.sh` → `train_qwen.py` | `checkpoint/Qwen25VL_7B{,_reverse}/S*_*/{checkpoint-1563,end}` |
| 5-task 추론+임베딩 | `extract_5_tasks/inf_qwen25vl.py` | `extract_5_tasks/result/qwen25vl_7b{,_reverse}/` |
| 5-task 채점 | `extract_5_tasks/acc_parsing.py --model qwen25vl_7b` (기존) | 〃 `/inference/accuracy` |
| 벤치마크 (line/grid/mmvp) | `extract_benchmark/ben_qwen25vl.py` | `extract_benchmark/result/qwen25vl_7b{,_reverse}/` |
| blind 8 task | `extract_benchmark/ben_qwen25vl_blind.py` + `acc_blind.py --model qwen25vl_7b_blind` (기존) | `extract_benchmark/result/qwen25vl_7b{,_reverse}_blind/` |
| 프로빙 | `probing/run_qwen25vl_probing.sh` (기존 probe_*.py 를 `--model qwen25vl_7b` 로 호출) | `probing/qwen25vl_7b{,_reverse}/` |
| 망각 eval_loss | `train/eval_loss/run_forgetting_qwen25vl.sh` → `eval_qwen25vl.py` | `train/eval_loss/results_qwen25vl_7b/` |

## 실행 순서

```bash
conda activate qwenvl

# 1) 학습 (GPU 0,1 = ZeRO-2, 글로벌 배치 16 → 3125 step, mid=1563). forward 끝나면 backward 이어서.
cd /home/hail/vlm3/train/qwen25vl_curriculum
GPU_IDS=0,1 bash run_all.sh 2>&1 | tee log_forward.txt && \
GPU_IDS=0,1 bash run_all_reverse.sh 2>&1 | tee log_reverse.txt

# 2) 5-task 추론 + 임베딩 → 채점
# 학습과 동시에 남는 GPU 2 에서 돌려도 된다. 이미 끝난 (ckpt, task) 는 자동으로 건너뛴다.
cd /home/hail/vlm3/extract_5_tasks
CUDA_VISIBLE_DEVICES=2 python inf_qwen25vl.py --model qwen25vl_7b
CUDA_VISIBLE_DEVICES=2 python inf_qwen25vl.py --model qwen25vl_7b_reverse
python acc_parsing.py --model qwen25vl_7b && python acc_parsing.py --model qwen25vl_7b_reverse

# 3) 벤치마크 + blind
cd /home/hail/vlm3/extract_benchmark
python ben_qwen25vl.py --run all
python ben_qwen25vl_blind.py --run all
python acc_blind.py --model qwen25vl_7b_blind && python acc_blind.py --model qwen25vl_7b_reverse_blind

# 4) 프로빙 (임베딩 준비 후)
cd /home/hail/vlm3/probing && bash run_qwen25vl_probing.sh

# 5) 망각 eval_loss
cd /home/hail/vlm3/train/eval_loss && bash run_forgetting_qwen25vl.sh
```

## 기존 모델과 맞춘 설정 / 다른 점

- **학습 하이퍼파라미터**: lr 1e-6, cosine, warmup 0.03, wd 0, 1 epoch, 글로벌 배치 16, bf16, max_len 2048, seed 42. vision tower 동결 + projector(merger) + LLM full FT (`freeze_vision_tower`, `tune_mm_mlp_adapter` 대응).
- **GPU**: 7B full FT 는 1장(96GB)에 안 들어가서 DeepSpeed 로 2장 사용. 기본은 ZeRO-2(`scripts/zero2.json`). `grad_accum = GLOBAL_BATCH / NUM_GPUS` 로 글로벌 배치 16 을 고정하므로 step 수(3125)와 mid(1563)가 llava 와 같다.
  - GPU 3장은 16 을 나눌 수 없어(15 또는 18만 가능) 쓰지 않는다. 배치를 다른 모델과 맞추는 쪽을 택했고, 남는 1장은 추론에 쓴다.
  - `GLOBAL_BATCH`/`MID_STEPS` 로 바꿀 수 있지만, 바꾸면 다른 4개 모델과 레시피가 달라진다.
  - GPU 지정은 `GPU_IDS`(→ `deepspeed --include localhost:...`). ⚠ deepspeed 런처가 `CUDA_VISIBLE_DEVICES` 를 덮어쓰므로 그 변수로는 GPU 를 고를 수 없다.

### 학습 설정 실측 (S3 = 최장 시퀀스 250 토큰, 최악 조건)

| 설정 | step당 | 글로벌 배치 | 초당 샘플 | 피크 GPU |
|---|---|---|---|---|
| 2 GPU ZeRO-2 ← **기본** | 6.41s | 16 | 2.50 | 87GiB |
| 2 GPU ZeRO-3 | 9.09s | 16 | 1.61 | 63GiB |
| 3 GPU ZeRO-2 | 4.52s | 15 | 3.26 | 63GiB |
| 3 GPU ZeRO-3 | 14.52s | 15 | 1.02 | 44GiB |
| 1 GPU ZeRO-3+CPU offload | ~23s | 16 | 0.70 | 22GiB |

GPU 간 연결이 PCIe(NVLink 없음)라서 ZeRO-3 의 파라미터 all-gather 가 비싸다. 그래서 GPU 를 3장으로 늘려도 ZeRO-3 는 오히려 느려진다. 학습 데이터는 전 stage 가 132~250 토큰으로 짧고 균일해서, S3 에서 잰 피크 메모리가 사실상 상한이다.

**예상 소요 시간** (stage당 5만 샘플 + start/mid/end eval 3회 + 체크포인트 2회 저장)

| 구간 | 시간 |
|---|---|
| stage 1개 | 약 5.9시간 |
| 한 방향 (5 stage) | 약 30시간 |
| forward + backward | 약 2.5일 |

**디스크**: 체크포인트 16GB × 2(mid+end) × 5 stage × 2 방향 = 320GB, 임베딩·결과 약 11GB, 베이스 모델 16GB → 합계 약 350GB. 루트 디스크 여유 2.9TB 로 충분하다.

- **gradient clipping**: `zero2.json`/`zero3.json` 모두 `gradient_clipping: auto` → HF `max_grad_norm=1.0` 적용. 원본 llava `scripts/zero2.json` 에는 이 키가 없어 mipha 는 클리핑 없이 학습됐다. 맞추고 싶으면 이 키를 지우면 된다.
- **mid 체크포인트**: HF `save_strategy` 대신 콜백에서 `step 1563` 에 모델 가중치+processor 만 저장 (optimizer state 저장 안 함 → 폴더당 ~16GB).
- **eval_loss**: mipha 처럼 각 stage 의 start/mid/end 에 held-out 5000 개로 eval (`EVAL_DATA_PATH`). `end/stage_eval_loss.json` 에 요약. 끄려면 `EVAL_AT_START=False` 등.
- **S1 은 nohex 사용 (중요)**: S1 Color Similarity 는 학습·검증·추론 **전부** 프롬프트에서 hex 코드를 뺀 nohex 버전을 쓴다. hex 프롬프트는 `(Left : #2800ff, Right : #0028ff)` 처럼 정답 계산에 필요한 정보를 그대로 흘려서, 모델이 이미지를 보지 않고도 풀 수 있다.
  | 용도 | 파일 |
  |---|---|
  | 학습 | `vlm_training/.../dataset/rebuttal_nohex/similarity_non_flamingo_nohex.json` (기존) |
  | 검증(eval_loss) | `vlm_validation/.../dataset/rebuttal_nohex/color_similarity_non_flamingo_nohex.json` (신규) |
  | 5-task 추론 | `vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_nohex/dataset_nohex.json` (신규) |

  검증셋·추론셋 nohex 는 `make_s1_nohex.py` 로 생성했다 (원본 불변, 새 파일만 작성). 변환 규칙은 기존 `train/s1_nohex_retrain/make_s1_prompts_nohex.py` 와 동일하게 `(Left : #xxxxxx, Right : #xxxxxx)` 괄호 구절만 제거하며, 실제 hex 값이 없는 "using their hex codes" 문장은 기존 nohex 학습셋과 맞추기 위해 남겨둔다. 이미지와 정답은 손대지 않는다.
  ⚠ 기존 llava/VILA-3B 는 본 체인이 hex 이고 nohex 는 `S1_Similarity_nohex` 라는 별도 분기였다. Qwen 은 본 체인 자체가 nohex 이므로 S2 이후가 nohex S1 위에 쌓인다 — S1 이후 시점은 다른 모델과 직접 비교할 때 이 차이를 감안해야 한다.
- **프롬프트**: LLaVA 포맷 질문에서 `<image>` 태그를 제거하고 Qwen chat template (`system: You are a helpful assistant.` 기본) 에 이미지 1장 + 텍스트로 넣는다. S2 ishihara 학습셋에 `<image>` 태그가 없어도 항상 이미지를 넣는다.
- **loss 마스킹**: 답변 텍스트 + `<|im_end|>` 만 학습 (프롬프트/시스템/이미지 토큰은 -100).
- **해상도**: `min_pixels=4*28*28`, `max_pixels=1024*28*28`. 224px → 64 토큰, 336px → 144 토큰, 2000px(counting_grid) → ≤1024 토큰.
- **임베딩 지점**: `LAYERS=[0,7,14,21,28]` (LLM 28층, hidden 3584; llava 의 0/8/16/24/32 에 대응) + `proj_feats` = `visual(...)`(merger 출력) 의 이미지 토큰 평균 (hidden 3584). `.pt` 스키마는 기존과 동일하므로 probing 코드 수정 없음.
- **생성**: greedy(temperature 0), `<|im_end|>` 에서 정지, 5-task/벤치 512 토큰, blind 64 토큰. HF `generate` 는 sequences 에 프롬프트를 포함하므로 잘라서 디코딩.
- **결과 폴더 이름**: `qwen25vl_7b`, `qwen25vl_7b_reverse` (+`_blind`). 집계 스크립트(`tools/export_csv.py` 의 `MODELS`, `probing/probe_expB_s4.py` 등)에 이 이름을 추가해야 표/그림에 포함된다.
