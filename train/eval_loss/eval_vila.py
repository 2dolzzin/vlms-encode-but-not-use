#!/usr/bin/env python
# eval_vila.py
# ─────────────────────────────────────────────────────────────────────────────
# VILA(3B/8B) 용 eval_loss 측정기. 무수정 VILA 의 llava/train/train.py 를 몽키패치로
# 가로채서 "한 개 체크포인트"를 로드 → 5개 task eval set 전부에 대해 eval_loss 를
# 찍는다. 학습은 0 스텝(평가만), 체크포인트 파일은 절대 건드리지 않는다.
#
# llava/mipha 판과의 차이(=VILA 특성):
#   - make_supervised_data_module(tokenizer, data_args, training_args)  ← 3인자
#   - LazySupervisedDataset(data_path, image_folder, tokenizer, data_args, training_args)
#   - Trainer = LLaVATrainer (VILA 의 것), callbacks 인자로 받음
#   - train_mem.py 가 train() 을 mock.patch 3종 컨텍스트에서 부르므로 그대로 재현
#   - 데이터는 data_mixture 레지스트리지만, eval 셋은 콜백에서 json 경로로 직접 빌드
#     (레지스트리 등록 불필요). 학습 train_dataset 은 더미(--data_mixture s0_logmar).
#
# 필요한 env:
#   EVAL_SETS_JSON / EVAL_OUT_JSON / EVAL_MODEL_TAG
# 런치는 run_forgetting_vila.sh 가 train_mem 과 동일한 torchrun+deepspeed 로 한다.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json

# VILA 패키지 경로 보장 (이 'llava' 는 NVIDIA VILA 의 것 — LLaVA 와 패키지명만 같음)
VILA_DIR = "/home/hail/vlm3/Model/VILA"
if VILA_DIR not in sys.path:
    sys.path.insert(0, VILA_DIR)

from unittest import mock
import llava.train.train as T
from llava.data.dataset import LazySupervisedDataset
from llava.train.transformer_normalize_monkey_patch import patched_normalize
from transformers import TrainerCallback

_EVAL_SETS = json.loads(os.environ["EVAL_SETS_JSON"])   # [["S0","path"], ...]
_OUT_JSON = os.environ["EVAL_OUT_JSON"]


def _p(msg):
    """rank0 만 출력 (VILA train.py 엔 rank0_print 가 없어 직접 구현)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    print(msg, flush=True)


# ── 1) tokenizer/data_args/training_args 캡처 ──
_CAPTURED = {}
_orig_make = T.make_supervised_data_module


def _make_capture(tokenizer, data_args, training_args):
    modules = _orig_make(tokenizer=tokenizer, data_args=data_args,
                         training_args=training_args)
    _CAPTURED["tokenizer"] = tokenizer
    _CAPTURED["data_args"] = data_args
    _CAPTURED["training_args"] = training_args
    return modules


T.make_supervised_data_module = _make_capture


# ── 2) deepspeed init 이후(on_train_begin) 5개 eval 후 즉시 종료 ──
class _ForgettingEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        tok = _CAPTURED["tokenizer"]
        dargs = _CAPTURED["data_args"]
        targs = _CAPTURED["training_args"]
        results = {}
        for name, path in _EVAL_SETS:
            _p(f"\n[eval_vila] === {name} eval: {path} ===")
            ds = LazySupervisedDataset(
                data_path=path, image_folder="/", tokenizer=tok,
                data_args=dargs, training_args=targs)
            metrics = self.trainer.evaluate(eval_dataset=ds,
                                            metric_key_prefix=f"eval_{name}")
            loss = metrics.get(f"eval_{name}_loss")
            results[name] = loss
            _p(f"[eval_vila] {name} eval_loss = {loss}  (n={len(ds)})")

        if self.trainer.is_world_process_zero():
            os.makedirs(os.path.dirname(_OUT_JSON), exist_ok=True)
            with open(_OUT_JSON, "w") as f:
                json.dump({
                    "model": os.environ.get("EVAL_MODEL_TAG", ""),
                    "model_path": args.output_dir,  # 참고용(버리는 temp)
                    "losses": results,
                }, f, indent=2)
            _p(f"[eval_vila] saved → {_OUT_JSON}")

        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        os._exit(0)


_OrigTrainer = T.LLaVATrainer


class _EvalTrainer(_OrigTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(_ForgettingEvalCallback(self))


T.LLaVATrainer = _EvalTrainer

# 안전장치: 종료 실패 시에도 체크포인트 안 덮어쓰도록 저장 함수 무력화
T.safe_save_model_for_hf_trainer = lambda *a, **k: None


# ── train_mem.py 의 mock.patch 3종 재현 (이미지 normalize + BatchSamplerShard) ──
def _bss_len(self):
    return len(self.batch_sampler)


def _bss_iter(self):
    return self.batch_sampler.__iter__()


if __name__ == "__main__":
    with (
        mock.patch('transformers.image_processing_utils.normalize', new=patched_normalize),
        mock.patch('accelerate.data_loader.BatchSamplerShard.__len__', new=_bss_len),
        mock.patch('accelerate.data_loader.BatchSamplerShard.__iter__', new=_bss_iter),
    ):
        T.train()
