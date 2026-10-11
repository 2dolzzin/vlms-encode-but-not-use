#!/usr/bin/env python
# train_with_eval.py
# mipha/train/train.py 를 "무수정"으로 쓰면서 eval_loss를 11시점(start/mid/end)에 찍는 래퍼.
#
# 두 가지를 몽키패치로 끼워넣음 (train.py는 안 건드림):
#   1) make_supervised_data_module → EVAL_DATA_PATH 의 held-out 데이터로 eval_dataset 생성
#   2) MiphaTrainer → start(step0)+mid+end 에서 trainer.evaluate() 호출하는 콜백 추가
#      (HF evaluation_strategy로는 mid 1563 과 end 3125 를 동시에 못 맞춰서 콜백으로 제어)
#
# 사용(.sh): deepspeed ... train_with_eval.py <train.py 인자 그대로>  + evaluation_strategy "no"
#   export EVAL_DATA_PATH=...        eval 데이터 (없으면 eval 안 함)
#   export EVAL_AT_STEPS="1563"      mid 시점(들). 쉼표로 여러개 가능. 기본 1563(=save_steps)
#   export EVAL_AT_START="1"         학습 전(step0=stage 시작점) eval 여부. 기본 on

import os
import sys

# train.py는 Model/llava-phi 안 mipha 패키지 → import 가능하게 경로 보장
LLAVA_PHI = "/home/hail/vlm3/Model/llava-phi"
if LLAVA_PHI not in sys.path:
    sys.path.insert(0, LLAVA_PHI)

import mipha.train.train as T
from mipha.train.train import LazySupervisedDataset
from transformers import TrainerCallback

EVAL_AT_STEPS = set(int(s) for s in os.environ.get("EVAL_AT_STEPS", "1563").split(",") if s.strip())
EVAL_AT_START = os.environ.get("EVAL_AT_START", "1") == "1"
_HAS_EVAL = bool(os.environ.get("EVAL_DATA_PATH", "").strip())


# ── 1) eval_dataset 주입 ──
_orig_make = T.make_supervised_data_module


def _make_with_eval(tokenizer, data_args):
    modules = _orig_make(tokenizer=tokenizer, data_args=data_args)
    eval_path = os.environ.get("EVAL_DATA_PATH", "").strip()
    if eval_path:
        modules["eval_dataset"] = LazySupervisedDataset(
            tokenizer=tokenizer, data_path=eval_path, data_args=data_args)
        T.rank0_print(f"[train_with_eval] eval_dataset 로드: {eval_path} "
                      f"(n={len(modules['eval_dataset'])})")
    else:
        T.rank0_print("[train_with_eval] EVAL_DATA_PATH 없음 → eval 미사용")
    return modules


T.make_supervised_data_module = _make_with_eval


# ── 2) 11시점 eval 콜백 (start / mid / end) ──
class _StageEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self._done = set()

    def _do_eval(self, tag, step):
        if step in self._done:
            return
        self._done.add(step)
        T.rank0_print(f"\n[train_with_eval] === eval @ {tag} (step {step}) ===")
        self.trainer.evaluate()   # eval_loss를 log_history + wandb에 기록

    def on_train_begin(self, args, state, control, **kwargs):
        if EVAL_AT_START:
            self._do_eval("start", 0)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in EVAL_AT_STEPS:
            self._do_eval("mid", state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self._do_eval("end", state.global_step)


_OrigTrainer = T.MiphaTrainer


class _MiphaTrainerWithEval(_OrigTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _HAS_EVAL:
            self.add_callback(_StageEvalCallback(self))

    def log(self, logs):
        # transformers 4.38.2 버그 우회: DeepSpeed grad_norm 이 Tensor 라서
        # log_history 에 그대로 들어가 save_to_json 시 직렬화 실패.
        # log() 단계에서 모든 Tensor 를 float 로 변환해 둔다.
        import torch
        for k, v in list(logs.items()):
            if isinstance(v, torch.Tensor):
                logs[k] = v.item() if v.numel() == 1 else v.tolist()
        super().log(logs)


T.MiphaTrainer = _MiphaTrainerWithEval


if __name__ == "__main__":
    T.train()
