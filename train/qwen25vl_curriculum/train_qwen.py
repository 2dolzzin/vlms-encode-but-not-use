#!/usr/bin/env python
# train_qwen.py
# Qwen2.5-VL-7B 커리큘럼 1 stage 학습 (HF Trainer + DeepSpeed ZeRO-2).
# llava/mipha 의 train.py + train_with_eval.py 역할을 하나로 합침:
#   - vision tower 동결, merger(projector) + LLM 전체 full fine-tune
#   - lr 1e-6 / cosine / warmup 0.03 / 1 epoch / global batch 16 (= 3125 step, 50k 샘플)
#   - mid(step 1563) 에 모델 가중치만 저장 → {mid_save_root}/checkpoint-1563
#   - end 에 {output_dir} 에 저장 (= OUT_DIR/end)
#   - EVAL_DATA_PATH 가 있으면 start / mid / end 에 eval_loss 를 기록 (mipha train_with_eval 과 동일)
#
# 저장물은 모두 "model + processor(tokenizer/preprocessor/chat_template)" 로,
# inf_qwen25vl.py 등이 from_pretrained 로 바로 읽을 수 있다. (optimizer state 는 저장 안 함)

import os
import sys
import json
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_data import (DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, DataCollatorForQwenVL,  # noqa: E402
                       LazySupervisedDataset, apply_freeze, load_model, load_processor)


def rank0_print(*a):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*a, flush=True)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/home/hail/vlm3/Model/Qwen2.5-VL-7B-Instruct")
    freeze_vision_tower: bool = field(default=True)
    tune_merger: bool = field(default=True)       # llava tune_mm_mlp_adapter
    freeze_llm: bool = field(default=False)       # llava freeze_backbone
    attn_implementation: str = field(default="sdpa")


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    eval_data_path: Optional[str] = field(default=None)   # 없으면 env EVAL_DATA_PATH
    image_folder: str = field(default="/")
    model_max_length: int = field(default=2048)
    min_pixels: int = field(default=DEFAULT_MIN_PIXELS)
    max_pixels: int = field(default=DEFAULT_MAX_PIXELS)


@dataclass
class StageArguments:
    mid_save_root: Optional[str] = field(default=None,
        metadata={"help": "mid 체크포인트를 {mid_save_root}/checkpoint-{step} 에 저장. 없으면 output_dir 상위."})
    mid_save_steps: str = field(default="1563", metadata={"help": "쉼표구분 step 들"})
    eval_at_start: bool = field(default=True)
    eval_at_mid: bool = field(default=True)
    eval_at_end: bool = field(default=True)


class StageCallback(TrainerCallback):
    """mid 저장 + start/mid/end eval. (HF save_strategy 로는 mid/end 를 동시에 못 맞춰 콜백으로 제어)"""

    def __init__(self, trainer, processor, mid_steps, mid_root, do_eval, ev_start, ev_mid, ev_end):
        self.trainer = trainer
        self.processor = processor
        self.mid_steps = set(mid_steps)
        self.mid_root = mid_root
        self.do_eval = do_eval
        self.ev_start, self.ev_mid, self.ev_end = ev_start, ev_mid, ev_end
        self._evaled = set()

    def _eval(self, tag, step):
        if not self.do_eval or (tag, step) in self._evaled:
            return
        self._evaled.add((tag, step))
        rank0_print(f"\n[train_qwen] === eval @ {tag} (step {step}) ===")
        m = self.trainer.evaluate(metric_key_prefix=f"eval_{tag}")
        rank0_print(f"[train_qwen] eval_{tag}_loss = {m.get(f'eval_{tag}_loss')}")

    def _save(self, path):
        rank0_print(f"[train_qwen] save → {path}")
        self.trainer.save_model(path)      # deepspeed 라도 rank0 가 full weight 저장 (zero2)
        if self.trainer.is_world_process_zero():
            self.processor.save_pretrained(path)

    def on_train_begin(self, args, state, control, **kw):
        if self.ev_start:
            self._eval("start", 0)

    def on_step_end(self, args, state, control, **kw):
        if state.global_step in self.mid_steps:
            self._save(os.path.join(self.mid_root, f"checkpoint-{state.global_step}"))
            if self.ev_mid:
                self._eval("mid", state.global_step)

    def on_train_end(self, args, state, control, **kw):
        if self.ev_end:
            self._eval("end", state.global_step)


class QwenTrainer(Trainer):
    def log(self, logs, *a, **kw):
        # deepspeed grad_norm 이 Tensor 로 들어와 trainer_state 직렬화가 깨지는 것 방지
        for k, v in list(logs.items()):
            if isinstance(v, torch.Tensor):
                logs[k] = v.item() if v.numel() == 1 else v.tolist()
        return super().log(logs, *a, **kw)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, StageArguments, TrainingArguments))
    margs, dargs, sargs, targs = parser.parse_args_into_dataclasses()

    if targs.gradient_checkpointing and targs.gradient_checkpointing_kwargs is None:
        targs.gradient_checkpointing_kwargs = {"use_reentrant": False}

    rank0_print(f"[train_qwen] model={margs.model_name_or_path}")
    rank0_print(f"[train_qwen] data={dargs.data_path}")

    processor = load_processor(margs.model_name_or_path, dargs.min_pixels, dargs.max_pixels)
    model = load_model(margs.model_name_or_path, dtype=torch.bfloat16 if targs.bf16 else torch.float32,
                       attn_implementation=margs.attn_implementation)
    model.config.use_cache = False

    n_train, n_total = apply_freeze(model, margs.freeze_vision_tower, margs.tune_merger, margs.freeze_llm)
    rank0_print(f"[train_qwen] trainable {n_train/1e9:.3f}B / {n_total/1e9:.3f}B params "
                f"(vision frozen={margs.freeze_vision_tower}, merger={margs.tune_merger}, llm frozen={margs.freeze_llm})")

    train_ds = LazySupervisedDataset(dargs.data_path, processor, dargs.model_max_length, dargs.image_folder)
    eval_path = dargs.eval_data_path or os.environ.get("EVAL_DATA_PATH", "").strip() or None
    eval_ds = LazySupervisedDataset(eval_path, processor, dargs.model_max_length, dargs.image_folder) if eval_path else None
    rank0_print(f"[train_qwen] train n={len(train_ds)}  eval n={len(eval_ds) if eval_ds else 0} ({eval_path})")

    collator = DataCollatorForQwenVL(processor.tokenizer)
    trainer = QwenTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                          data_collator=collator, processing_class=processor)

    mid_root = sargs.mid_save_root or os.path.dirname(os.path.normpath(targs.output_dir))
    mid_steps = [int(s) for s in sargs.mid_save_steps.split(",") if s.strip()]
    trainer.add_callback(StageCallback(trainer, processor, mid_steps, mid_root, eval_ds is not None,
                                       sargs.eval_at_start, sargs.eval_at_mid, sargs.eval_at_end))

    world = max(1, targs.world_size)
    gb = targs.per_device_train_batch_size * targs.gradient_accumulation_steps * world
    rank0_print(f"[train_qwen] global batch = {gb}  → total steps ≈ {math.ceil(len(train_ds)/gb)}  mid={mid_steps}")

    trainer.train()
    if torch.cuda.is_available():
        rank0_print(f"[train_qwen] peak GPU mem (rank0): allocated {torch.cuda.max_memory_allocated()/2**30:.1f} GiB, "
                    f"reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GiB")

    # end 저장 (= OUT_DIR/end)
    trainer.save_state()
    trainer.save_model(targs.output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(targs.output_dir)
        # eval_loss 요약 (start/mid/end) 를 따로 남겨 두면 나중에 표 만들 때 편함
        evals = {k: v for e in trainer.state.log_history for k, v in e.items() if k.startswith("eval_") and k.endswith("_loss")}
        with open(os.path.join(targs.output_dir, "stage_eval_loss.json"), "w") as f:
            json.dump(evals, f, indent=2)
        rank0_print(f"[train_qwen] done → {targs.output_dir}  eval={evals}")


if __name__ == "__main__":
    main()
