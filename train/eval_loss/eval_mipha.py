#!/usr/bin/env python
# eval_mipha.py
# ─────────────────────────────────────────────────────────────────────────────
# Mipha-3B(phi2) 용 eval_loss 측정기. 무수정 mipha/train/train.py 를 몽키패치로
# 가로채서 "한 개 체크포인트"를 로드 → 5개 task eval set 전부에 대해 eval_loss 를
# 찍는다. 학습은 0 스텝(평가만), 체크포인트 파일은 절대 건드리지 않는다.
#
# 동작:
#   1) make_supervised_data_module 몽키패치 → tokenizer/data_args 캡처
#   2) on_train_begin(=deepspeed init 직후) 에서 EVAL_SETS_JSON 5개를
#      LazySupervisedDataset 으로 만들어 trainer.evaluate() 5번 → eval_loss 수집
#   3) 저장 후 즉시 os._exit(0) (학습 스텝/모델 저장 없이) → 체크포인트 보호
#
# 필요한 env:
#   EVAL_SETS_JSON : '[["S0","/abs/s0_eval.json"], ...]'  (순서 보존 list)
#   EVAL_OUT_JSON  : 결과 저장 경로
#   EVAL_MODEL_TAG : 결과 JSON 의 "model" 필드용 태그 (선택)
#
# 런치는 run_forgetting_mipha.sh 가 train.py(mipha) 와 동일한 deepspeed 플래그로 한다.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json

# mipha 패키지 경로 보장
LLAVA_PHI = "/home/hail/vlm3/Model/llava-phi"
if LLAVA_PHI not in sys.path:
    sys.path.insert(0, LLAVA_PHI)

import mipha.train.train as T
from mipha.train.train import LazySupervisedDataset
from transformers import TrainerCallback

# 순서 보존을 위해 list-of-pairs 로 받는다.
_EVAL_SETS = json.loads(os.environ["EVAL_SETS_JSON"])   # [["S0","path"], ...]
_OUT_JSON = os.environ["EVAL_OUT_JSON"]

# ── 1) tokenizer/data_args 캡처 (학습 데이터셋은 원래대로 생성) ──
_CAPTURED = {}
_orig_make = T.make_supervised_data_module


def _make_capture(tokenizer, data_args):
    modules = _orig_make(tokenizer=tokenizer, data_args=data_args)
    _CAPTURED["tokenizer"] = tokenizer
    _CAPTURED["data_args"] = data_args
    return modules


T.make_supervised_data_module = _make_capture


# ── 2) deepspeed init 이후(on_train_begin) 5개 eval 수행 후 즉시 종료 ──
class _ForgettingEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        tok = _CAPTURED["tokenizer"]
        dargs = _CAPTURED["data_args"]
        results = {}
        for name, path in _EVAL_SETS:
            T.rank0_print(f"\n[eval_mipha] === {name} eval: {path} ===")
            ds = LazySupervisedDataset(tokenizer=tok, data_path=path, data_args=dargs)
            metrics = self.trainer.evaluate(eval_dataset=ds,
                                            metric_key_prefix=f"eval_{name}")
            loss = metrics.get(f"eval_{name}_loss")
            results[name] = loss
            T.rank0_print(f"[eval_mipha] {name} eval_loss = {loss}  (n={len(ds)})")

        # rank0 만 저장
        if self.trainer.is_world_process_zero():
            os.makedirs(os.path.dirname(_OUT_JSON), exist_ok=True)
            with open(_OUT_JSON, "w") as f:
                json.dump({
                    "model": os.environ.get("EVAL_MODEL_TAG", ""),
                    "model_path": args.output_dir,  # 참고용(버리는 temp)
                    "losses": results,
                }, f, indent=2)
            T.rank0_print(f"[eval_mipha] saved → {_OUT_JSON}")

        # 다른 rank 와 동기화 후, 학습/저장 전혀 안 하고 깔끔히 종료
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        os._exit(0)


_OrigTrainer = T.MiphaTrainer


class _EvalTrainer(_OrigTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(_ForgettingEvalCallback(self))

    def log(self, logs):
        # Tensor→float 우회 (원본과 동일)
        import torch
        for k, v in list(logs.items()):
            if isinstance(v, torch.Tensor):
                logs[k] = v.item() if v.numel() == 1 else v.tolist()
        super().log(logs)


T.MiphaTrainer = _EvalTrainer

# 안전장치: 종료 실패 시에도 체크포인트 안 덮어쓰도록 저장 함수 무력화
T.safe_save_model_for_hf_trainer = lambda *a, **k: None


if __name__ == "__main__":
    T.train()
