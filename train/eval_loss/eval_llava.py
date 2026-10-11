import os
import sys
import json

# llava 패키지 경로 보장
LLAVA_DIR = "/home/hail/vlm3/Model/LLaVA"
if LLAVA_DIR not in sys.path:
    sys.path.insert(0, LLAVA_DIR)

import llava.train.train as T
from llava.train.train import LazySupervisedDataset
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
            T.rank0_print(f"\n[eval_llava] === {name} eval: {path} ===")
            ds = LazySupervisedDataset(tokenizer=tok, data_path=path, data_args=dargs)
            metrics = self.trainer.evaluate(eval_dataset=ds,
                                            metric_key_prefix=f"eval_{name}")
            loss = metrics.get(f"eval_{name}_loss")
            results[name] = loss
            T.rank0_print(f"[eval_llava] {name} eval_loss = {loss}  (n={len(ds)})")

        # rank0 만 저장
        if self.trainer.is_world_process_zero():
            os.makedirs(os.path.dirname(_OUT_JSON), exist_ok=True)
            with open(_OUT_JSON, "w") as f:
                json.dump({
                    "model": os.environ.get("EVAL_MODEL_TAG", ""),
                    "model_path": args.output_dir,  # 참고용(버리는 temp)
                    "losses": results,
                }, f, indent=2)
            T.rank0_print(f"[eval_llava] saved → {_OUT_JSON}")

        # 다른 rank 와 동기화 후, 학습/저장 전혀 안 하고 깔끔히 종료
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        # 학습 스텝/end-of-train 저장 로직을 타지 않도록 즉시 프로세스 종료
        os._exit(0)


_OrigTrainer = T.LLaVATrainer


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


T.LLaVATrainer = _EvalTrainer

# 안전장치: 혹시라도 종료가 안 됐을 때 체크포인트를 덮어쓰지 않도록 저장 함수 무력화
T.safe_save_model_for_hf_trainer = lambda *a, **k: None


if __name__ == "__main__":
    # train_mem.py 와 동일하게 flash_attention_2 로 진입
    T.train(attn_implementation="flash_attention_2")
