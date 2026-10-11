import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset, Sampler
from transformers import Trainer, GenerationConfig
from transformers.trainer import has_length
from PIL import Image

from open_flamingo import create_model_and_transforms

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IGNORE_INDEX = -100

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ──────────────────────────────────────────────
# Argument dataclasses
# ──────────────────────────────────────────────
@dataclass
class ModelArguments:
    vision_encoder_path: str = field(default="ViT-L-14")
    vision_encoder_pretrained: str = field(default="openai")
    lm_path: str = field(default="anas-awadalla/mpt-1b-redpajama-200b")
    tokenizer_path: Optional[str] = field(default=None)
    cross_attn_every_n_layers: int = field(default=1)
    freeze_vision_encoder: bool = field(default=True)
    # 기본 True = 임베딩 동결 → perceiver + cross-attn 딱 2개만 학습
    freeze_lm_embeddings: bool = field(default=True)
    # "adapter" = OF 철학(2개 모듈만) / "fft" = vision 빼고 전부(LLM 포함)
    train_mode: str = field(default="adapter")
    pretrained_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained OpenFlamingo .pt checkpoint to load before training."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training JSON."})
    image_folder: Optional[str] = field(default=None)
    eval_data_path: Optional[str] = field(default=None)
    eval_data_paths: Optional[str] = field(
        default=None,
        metadata={"help": "JSON string mapping stage names to eval data paths (multi-task loss matrix)."},
    )
    lazy_preprocess: bool = field(default=True)
    image_aspect_ratio: str = field(default="pad")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=2048)
    group_by_modality_length: bool = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)


# ──────────────────────────────────────────────
# Flamingo wrapper for HF Trainer
# ──────────────────────────────────────────────
class FlamingoForTraining(torch.nn.Module):
    def __init__(self, flamingo_model, config=None):
        super().__init__()
        self.flamingo = flamingo_model
        if config is not None:
            self.config = config
        else:
            self.config = self.flamingo.lang_encoder.config

        if hasattr(self.flamingo.lang_encoder, "generation_config"):
            self.generation_config = self.flamingo.lang_encoder.generation_config
        else:
            self.generation_config = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        images: torch.Tensor = None,
        **kwargs,
    ):
        if images is not None:
            if images.ndim == 4:
                vision_x = images.unsqueeze(1).unsqueeze(1)
            elif images.ndim == 6:
                vision_x = images
            else:
                raise ValueError(f"Unexpected images ndim={images.ndim}")
        else:
            vision_x = None

        output = self.flamingo(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output

    def state_dict(self, *args, **kwargs):
        return self.flamingo.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.flamingo.load_state_dict(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.flamingo.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.flamingo.named_parameters(*args, **kwargs)

    def modules(self):
        return self.flamingo.modules()

    def named_modules(self, *args, **kwargs):
        return self.flamingo.named_modules(*args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # No-op: OpenFlamingo applies its own gradient checkpointing on the
        # FlamingoLayers (gated cross-attn + decoder) at model construction
        # time via the `gradient_checkpointing` flag. MosaicGPT does not
        # support HF's native checkpointing, so we must NOT delegate here.
        rank0_print(
            "[INFO] gradient_checkpointing handled by OpenFlamingo at build time; "
            "HF Trainer call is a no-op"
        )

    def gradient_checkpointing_disable(self):
        pass

    def train(self, mode=True):
        self.flamingo.train(mode)
        return self

    def eval(self):
        self.flamingo.eval()
        return self

    def save_pretrained(self, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.flamingo.state_dict(), os.path.join(output_dir, "flamingo_model.pt"))
        self.config.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, path, flamingo_model):
        ckpt_path = os.path.join(path, "flamingo_model.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            flamingo_model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded flamingo checkpoint from {ckpt_path}")
        return cls(flamingo_model)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor,
        data_args: DataArguments,
    ):
        super().__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_args = data_args

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _build_prompt_and_labels(self, conversations):
        prompt = ""
        label_parts = []

        for conv in conversations:
            role = conv["from"]
            text = conv["value"]

            if role == "human":
                turn = f"USER: {text}\n"
                start = len(prompt)
                prompt += turn
                label_parts.append((start, len(prompt), False))
            elif role == "gpt":
                turn = f"ASSISTANT: {text}<|endofchunk|>\n"
                start = len(prompt)
                prompt += turn
                prefix = "ASSISTANT: "
                label_parts.append((start, start + len(prefix), False))
                label_parts.append((start + len(prefix), len(prompt), True))

        return prompt, label_parts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        conversations = sample["conversations"]

        prompt_text, label_parts = self._build_prompt_and_labels(conversations)

        tokenized = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        input_ids = tokenized.input_ids[0]

        labels = input_ids.clone()

        media_token_id = self.tokenizer.encode("<image>", add_special_tokens=False)[-1]
        labels[labels == media_token_id] = IGNORE_INDEX
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        self._mask_human_turns(input_ids, labels)

        data_dict = dict(input_ids=input_ids, labels=labels)

        if "image" in sample:
            image_path = os.path.join(self.data_args.image_folder, sample["image"])
            image = Image.open(image_path).convert("RGB")

            if self.data_args.image_aspect_ratio == "pad":
                image = self._expand2square(image)

            image_tensor = self.image_processor(image)
            if isinstance(image_tensor, dict):
                image_tensor = image_tensor["pixel_values"]
            if isinstance(image_tensor, list):
                image_tensor = image_tensor[0]
            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.tensor(image_tensor)

            data_dict["image"] = image_tensor
        else:
            data_dict["image"] = torch.zeros(3, 224, 224)

        valid = (labels != IGNORE_INDEX).sum().item()
        if valid == 0:
            print(f"[DBG] valid_labels=0 idx={i} has_image={'image' in sample}")

        return data_dict

    def _mask_human_turns(self, input_ids, labels):
        user_token_ids = self.tokenizer.encode("USER:", add_special_tokens=False)
        assistant_token_ids = self.tokenizer.encode("ASSISTANT:", add_special_tokens=False)

        ids_list = input_ids.tolist()
        i = 0
        while i < len(ids_list):
            if self._match_tokens(ids_list, i, user_token_ids):
                j = i
                while j < len(ids_list):
                    if self._match_tokens(ids_list, j, assistant_token_ids):
                        j += len(assistant_token_ids)
                        break
                    j += 1
                labels[i:j] = IGNORE_INDEX
                i = j
            else:
                i += 1

        if len(ids_list) > 0 and not self._match_tokens(ids_list, 0, user_token_ids):
            for k in range(len(ids_list)):
                if self._match_tokens(ids_list, k, user_token_ids):
                    labels[:k] = IGNORE_INDEX
                    break

    def _match_tokens(self, ids_list, pos, pattern):
        if pos + len(pattern) > len(ids_list):
            return False
        return ids_list[pos : pos + len(pattern)] == pattern

    def _expand2square(self, pil_img):
        width, height = pil_img.size
        if width == height:
            return pil_img
        size = max(width, height)
        result = Image.new(pil_img.mode, (size, size), (128, 128, 128))
        if width > height:
            result.paste(pil_img, (0, (size - height) // 2))
        else:
            result.paste(pil_img, ((size - width) // 2, 0))
        return result


# ──────────────────────────────────────────────
# Data collator
# ──────────────────────────────────────────────
@dataclass
class DataCollatorForFlamingo:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        max_len = self.tokenizer.model_max_length
        input_ids = input_ids[:, :max_len]
        labels = labels[:, :max_len]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [inst["image"] for inst in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


# ──────────────────────────────────────────────
# Data module factory — eval_data_paths(dict) 지원 (optional)
# ──────────────────────────────────────────────
def make_supervised_data_module(
    tokenizer,
    image_processor,
    data_args,
) -> Dict:
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_args=data_args,
    )

    eval_dataset = None
    if getattr(data_args, "eval_data_paths", None) is not None:
        try:
            paths_dict = json.loads(data_args.eval_data_paths)
            assert isinstance(paths_dict, dict)
            eval_dataset = {}
            for stage_name, p in paths_dict.items():
                if p and os.path.isfile(p):
                    rank0_print(f"[EVAL MATRIX] Loading {stage_name}: {p}")
                    eval_dataset[stage_name] = LazySupervisedDataset(
                        data_path=p,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        data_args=data_args,
                    )
                else:
                    rank0_print(f"[EVAL MATRIX] SKIP {stage_name}: path not found ({p})")
            if not eval_dataset:
                eval_dataset = None
        except Exception as e:
            rank0_print(f"[EVAL MATRIX] failed to parse eval_data_paths: {e}")
            eval_dataset = None

    if eval_dataset is None and data_args.eval_data_path is not None:
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            data_args=data_args,
        )

    data_collator = DataCollatorForFlamingo(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


# ──────────────────────────────────────────────
# Freeze 정책
# ──────────────────────────────────────────────
def apply_freeze_policy(flamingo_model, model_args):
    """
    train_mode == "adapter" (기본):
        OpenFlamingo 원래 철학. perceiver + gated cross-attn 만 학습.
        vision encoder, LLM 본체 동결. 임베딩은 freeze_lm_embeddings 로 제어(기본 동결).
    train_mode == "fft":
        vision encoder만 동결, 나머지 전부 학습 (LLaVA/Mipha/VILA 와 같은 선상).
    """
    mode = model_args.train_mode.lower()

    if mode == "fft":
        flamingo_model.requires_grad_(True)
        if model_args.freeze_vision_encoder:
            flamingo_model.vision_encoder.requires_grad_(False)
        rank0_print("[FREEZE] train_mode=fft (vision 동결, 나머지 전부 학습)")
        return

    # ── adapter (기본): 2개 모듈만 ──
    flamingo_model.requires_grad_(False)
    flamingo_model.perceiver.requires_grad_(True)
    flamingo_model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not model_args.freeze_lm_embeddings:
        # 주의: HF Trainer 경로엔 OF 원본 학습루프의 임베딩 grad 마스킹이 없으므로
        #       이 옵션을 켜면 전체 임베딩 테이블이 학습됨.
        flamingo_model.lang_encoder.get_input_embeddings().requires_grad_(True)
    # vision encoder 명시적 동결 보장
    flamingo_model.vision_encoder.requires_grad_(False)
    rank0_print(
        f"[FREEZE] train_mode=adapter (perceiver + gated cross-attn"
        f"{' + lm_embeddings' if not model_args.freeze_lm_embeddings else ''} 만 학습)"
    )


def _summarize_trainable(model):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (
            "perceiver" if "perceiver" in n
            else "gated_cross_attn" if "gated_cross_attn" in n
            else "embeddings" if "embed" in n
            else "vision_encoder" if "vision_encoder" in n
            else "lang_encoder(other)"
        )
        groups[key] = groups.get(key, 0) + p.numel()
    rank0_print("[FREEZE] 학습 대상 파라미터 그룹:")
    for k, v in sorted(groups.items()):
        rank0_print(f"          - {k}: {v:,}")


# ──────────────────────────────────────────────
# FlamingoTrainer (LLaVA/Mipha Trainer 패턴)
# ──────────────────────────────────────────────
def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]
    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]
    additional_batch = mm_megabatches[-1] + lang_megabatches[-1]
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))
    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    def __init__(self, batch_size, world_size, lengths=None, generator=None, group_by_modality=False):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class FlamingoTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        def _to_python(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
                return x.item() if x.numel() == 1 else x.tolist()
            if isinstance(x, np.generic):
                return x.item()
            return x
        logs = {k: _to_python(v) for k, v in logs.items()}
        return super().log(logs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if metrics is not None:
            cleaned = {}
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu()
                    cleaned[k] = v.item() if v.numel() == 1 else v.tolist()
                elif isinstance(v, np.generic):
                    cleaned[k] = v.item()
                else:
                    cleaned[k] = v
            metrics = cleaned
        super(FlamingoTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        inner_model = self.model
        if hasattr(inner_model, "flamingo"):
            lang_enc = inner_model.flamingo.lang_encoder
        elif hasattr(inner_model, "module") and hasattr(inner_model.module, "flamingo"):
            lang_enc = inner_model.module.flamingo.lang_encoder
        else:
            lang_enc = None
        if lang_enc is not None and getattr(lang_enc, "generation_config", None) is not None:
            try:
                gc_dict = lang_enc.generation_config.to_dict()
            except Exception:
                gc_dict = {}
            fixed = GenerationConfig.from_dict(gc_dict)
            fixed.do_sample = False
            fixed.temperature = 1.0
            fixed.top_p = 1.0
            fixed.top_k = 50
            lang_enc.generation_config = fixed
        super(FlamingoTrainer, self)._save(output_dir, state_dict)


# ──────────────────────────────────────────────
# Main train function
# ──────────────────────────────────────────────
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # 1. 모델 생성
    tokenizer_path = model_args.tokenizer_path or model_args.lm_path

    flamingo_model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=model_args.vision_encoder_path,
        clip_vision_encoder_pretrained=model_args.vision_encoder_pretrained,
        lang_encoder_path=model_args.lm_path,
        tokenizer_path=tokenizer_path,
        cross_attn_every_n_layers=model_args.cross_attn_every_n_layers,
        freeze_lm_embeddings=model_args.freeze_lm_embeddings,
        gradient_checkpointing=training_args.gradient_checkpointing,
    )

    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"

    # 2. 사전학습 체크포인트 로드 (S0 시작점 = OF pretrained, 이후 stage = 앞 stage 결과)
    if model_args.pretrained_checkpoint is not None:
        ckpt = torch.load(model_args.pretrained_checkpoint, map_location="cpu")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        flamingo_model.load_state_dict(state_dict, strict=False)
        rank0_print(f"[INFO] Loaded pretrained checkpoint from {model_args.pretrained_checkpoint}")

    # 3. freeze 정책 (기본: 어댑터 전용 = perceiver + cross-attn 만)
    apply_freeze_policy(flamingo_model, model_args)
    vt_trainable = sum(p.numel() for p in flamingo_model.vision_encoder.parameters() if p.requires_grad)
    rank0_print(f"[DBG] vision_encoder trainable params = {vt_trainable}")

    # 4. generation_config 안전 처리
    flamingo_model.lang_encoder.config.use_cache = False
    if hasattr(flamingo_model.lang_encoder, "generation_config") and flamingo_model.lang_encoder.generation_config is not None:
        flamingo_model.lang_encoder.generation_config.do_sample = False
        flamingo_model.lang_encoder.generation_config.temperature = 1.0
        flamingo_model.lang_encoder.generation_config.top_p = 1.0

    # 5. Wrapper
    model = FlamingoForTraining(flamingo_model)

    # 6. Gradient checkpointing
    # MosaicGPT(MPT) does NOT support HF's native gradient checkpointing, so we
    # use OpenFlamingo's mechanism: the `gradient_checkpointing=True` flag passed
    # to create_model_and_transforms marks the FlamingoLayers (gated cross-attn +
    # decoder) and the perceiver with `_use_gradient_checkpointing`, then we wrap
    # those marked submodules with torch's activation-checkpoint wrapper here.
    if training_args.gradient_checkpointing:
        import functools
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            CheckpointWrapper,
            apply_activation_checkpointing,
        )

        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, CheckpointWrapper),
        )
        rank0_print("[INFO] Applied OpenFlamingo activation checkpointing to marked layers")

        # Ensure checkpointed segments receive grad-requiring inputs even though
        # the input embeddings are frozen (adapter-only training).
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        flamingo_model.lang_encoder.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad
        )

    # 7. Dataset
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_args=data_args,
    )

    try:
        sample = data_module["train_dataset"][0]
        valid = (sample["labels"] != IGNORE_INDEX).sum().item()
        rank0_print(f"[DBG] first sample: input_ids len={len(sample['input_ids'])}, valid labels={valid}")
        if "image" in sample:
            rank0_print(f"[DBG] first sample image shape={sample['image'].shape}")
    except Exception as e:
        rank0_print(f"[DBG] failed to inspect first sample: {e}")

    # 8. Trainer
    trainer = FlamingoTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    _summarize_trainable(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    rank0_print(f"[DBG] trainable/total = {trainable}/{total}")

    # 9. Train
    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_state()

    # 10. Save final model
    flamingo_model.lang_encoder.config.use_cache = True

    if trainer.args.should_save:
        output_dir = training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            flamingo_model.state_dict(),
            os.path.join(output_dir, "flamingo_model.pt"),
        )
        tokenizer.save_pretrained(output_dir)
        flamingo_model.lang_encoder.config.save_pretrained(output_dir)
        rank0_print(f"[INFO] Model saved to {output_dir}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    rank0_print(f"[DBG] final trainable/total = {trainable}/{total}")


if __name__ == "__main__":
    train()
