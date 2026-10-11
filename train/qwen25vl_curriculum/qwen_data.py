# qwen_data.py
# Qwen2.5-VL 커리큘럼 학습 / eval_loss 공용 모듈.
#   - LLaVA 포맷 json (id / image / conversations[human,gpt]) → Qwen2.5-VL chat 입력
#   - LazySupervisedDataset : 질문(프롬프트) 토큰은 label -100, 답변 + <|im_end|> 만 학습
#   - DataCollator          : input_ids/labels 우측 패딩, pixel_values/image_grid_thw 는 concat
#   - load_model_and_processor / apply_freeze : 모델 로드 + (vision 동결, merger+LLM 학습)
#
# 기존 llava/mipha 학습과 맞춘 점:
#   * <image> 태그는 제거하고 항상 이미지를 1장 넣는다 (S2 ishihara 학습셋엔 태그가 없음).
#   * vision tower 동결, projector(=merger) + LLM 전체 학습 (tune_mm_mlp_adapter + freeze_vision_tower).
#   * model_max_length 로 truncation.

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100

# Qwen2.5-VL 동적 해상도 제어. 28px 패치 단위.
#   224x224 → 8x8=64 토큰, 336x336 → 144 토큰, 2000x2000(benchmark counting_grid) → 상한 1024 토큰
DEFAULT_MIN_PIXELS = 4 * 28 * 28
DEFAULT_MAX_PIXELS = 1024 * 28 * 28


def build_messages(question: str):
    """LLaVA 질문 문자열 → Qwen chat messages (이미지 1장 + 텍스트)."""
    q = question.replace("<image>", "").strip()
    return [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": q}]}]


def build_prompt(processor, question: str) -> str:
    return processor.apply_chat_template(build_messages(question),
                                         tokenize=False, add_generation_prompt=True)


def load_processor(path, min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(path, min_pixels=min_pixels, max_pixels=max_pixels)
    # 우측 패딩 (배치 학습/eval 용. generate 는 batch=1 이라 무관)
    proc.tokenizer.padding_side = "right"
    return proc


def load_model(path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None):
    from transformers import Qwen2_5_VLForConditionalGeneration
    kw = dict(torch_dtype=dtype, attn_implementation=attn_implementation)
    if device_map is not None:
        kw["device_map"] = device_map
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kw)


def get_visual(model):
    """transformers 버전에 따라 model.visual / model.model.visual 둘 다 대응."""
    v = getattr(model, "visual", None)
    if v is None:
        v = model.model.visual
    return v


def text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def apply_freeze(model, freeze_vision_tower=True, tune_merger=True, freeze_llm=False):
    """llava 의 freeze_vision_tower / tune_mm_mlp_adapter / freeze_backbone 에 대응.
    파라미터 이름 기준: '*.visual.*' 가 vision, 그 중 '*.merger.*' 가 projector."""
    n_train = n_total = 0
    for name, p in model.named_parameters():
        is_vis = ".visual." in f".{name}"
        is_merger = is_vis and ".merger." in f".{name}"
        if is_merger:
            p.requires_grad = bool(tune_merger)
        elif is_vis:
            p.requires_grad = not freeze_vision_tower
        else:
            p.requires_grad = not freeze_llm
        n_total += p.numel()
        n_train += p.numel() if p.requires_grad else 0
    return n_train, n_total


class LazySupervisedDataset(Dataset):
    """json 을 통째로 들고 있다가 __getitem__ 에서 이미지 로드 + 토크나이즈 (llava lazy_preprocess 와 동일)."""

    def __init__(self, data_path, processor, max_len=2048, image_folder="/"):
        with open(data_path) as f:
            self.records = json.load(f)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_len = max_len
        self.image_folder = image_folder
        self.im_end_ids = self.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.records)

    def _image_path(self, rec):
        p = rec["image"]
        return p if os.path.isabs(p) else os.path.join(self.image_folder, p)

    def __getitem__(self, i):
        rec = self.records[i]
        image = Image.open(self._image_path(rec)).convert("RGB")
        question = rec["conversations"][0]["value"]
        answer = rec["conversations"][1]["value"]

        prompt = build_prompt(self.processor, question)
        enc = self.processor(text=[prompt], images=[image], return_tensors="pt")
        prompt_ids = enc["input_ids"][0]

        # 답변은 텍스트 토크나이저로만 (경계 병합 이슈 없이 프롬프트/답변 구간을 정확히 분리)
        ans_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"] + self.im_end_ids
        ans_ids = torch.tensor(ans_ids, dtype=prompt_ids.dtype)

        input_ids = torch.cat([prompt_ids, ans_ids])
        labels = torch.cat([torch.full_like(prompt_ids, IGNORE_INDEX), ans_ids.clone()])
        if input_ids.numel() > self.max_len:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]

        return dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=enc["pixel_values"],          # (n_patches, patch_dim)
            image_grid_thw=enc["image_grid_thw"],      # (1, 3)
        )


class DataCollatorForQwenVL:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, batch):
        ids = [b["input_ids"] for b in batch]
        labs = [b["labels"] for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.pad_id)
        # 패딩 토큰이 실제 eos 와 같은 id 일 수 있어 길이 기반으로 다시 마스크
        for r, x in enumerate(ids):
            attention_mask[r, : x.numel()] = True
            attention_mask[r, x.numel():] = False
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=torch.cat([b["pixel_values"] for b in batch], dim=0),
            image_grid_thw=torch.cat([b["image_grid_thw"] for b in batch], dim=0),
        )
