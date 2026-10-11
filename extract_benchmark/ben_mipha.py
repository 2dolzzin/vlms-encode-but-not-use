import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# ben_mipha.py
# Mipha-3B(phi2) 커리큘럼 11시점 ckpt 로 일반 VLM 벤치마크(line_plot/counting_grid/mmvp)
# 추론 + 임베딩 추출 + 채점. (inf_mipha.py 를 forward/reverse CLI 로 일반화)
#
# 실행 (extract_benchmark/ 안에서):
#   python ben_mipha.py --run mipha          # forward
#   python ben_mipha.py --run mipha_reverse    # reverse
#   python ben_mipha.py --run mipha --eval_only

import sys
import re
import glob
import json
import argparse

import numpy as np
from sklearn.metrics import f1_score

import torch
from PIL import Image
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

# ── 모델 정체성 (Mipha) ───────────────────────────────────────
MIPHA_REPO = os.path.join(ROOT, "Model/llava-phi")
CONV_MODE  = "phi"
MODEL_BASE = None
MODEL_NAME = "mipha-phi2"
LAYERS = [0, 8, 16, 24, 32]
TOP_P          = 0.7
MAX_NEW_TOKENS = 512

RUNS = {
    "mipha":       dict(root=os.path.join(ROOT, "checkpoint/Mipha"),         out="mipha"),
    "mipha_reverse": dict(root=os.path.join(ROOT, "checkpoint/Mipha_reverse"), out="mipha_reverse"),
}
BASE_MODEL = os.path.join(ROOT, "Model/llava-phi/Mipha-3B-phi2")
STAGES = {
    "S0": "S0_LogMAR-phi2", "S1": "S1_Similarity-phi2", "S2": "S2_Ishihara-phi2",
    "S3": "S3_Naming-phi2", "S4": "S4_Depth-phi2",
}

DATASETS = {
    "line_plot":     "/home/hail/HDD/vlm_inference/benchmark/Task_lowlevel/line_plot/dataset.json",
    "counting_grid": "/home/hail/HDD/vlm_inference/benchmark/Task_lowlevel/counting_grid/dataset.json",
    "mmvp":          "/home/hail/HDD/vlm_inference/benchmark/MMVP/dataset.json",
}


def build_ckpts(root):
    ck = {"S0_start": BASE_MODEL}
    for sid, dname in STAGES.items():
        sdir = os.path.join(root, dname)
        mids = sorted(glob.glob(os.path.join(sdir, "checkpoint-*")))
        if len(mids) > 1:
            print(f"[warn] {sdir} 에 checkpoint-* 여러 개 → {mids[-1]} 사용")
        if mids:
            ck[f"{sid}_mid"] = mids[-1]
        end = os.path.join(sdir, "end")
        if os.path.isdir(end):
            ck[f"{sid}_end"] = end
    return {k: v for k, v in ck.items() if k == "S0_start" or os.path.exists(v)}


def load_dataset(path):
    with open(path) as f:
        cases = json.load(f)
    return [{
        "id": str(c.get("id", "")),
        "image_path": c["image"],
        "question": c["conversations"][0]["value"],
        "answer": c["conversations"][1]["value"],
    } for c in cases]


# ── Mipha loader (inf_mipha.py 와 동일) ───────────────────────
class MiphaInferExtract:
    def __init__(self, ckpt_path):
        sys.path.insert(0, MIPHA_REPO)
        from mipha.model.builder import load_pretrained_model
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            ckpt_path, MODEL_BASE, MODEL_NAME)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

    @torch.inference_mode()
    def run_one(self, image_path, question, layers, temperature):
        from mipha.mm_utils import (KeywordsStoppingCriteria, process_images,
                                     tokenizer_image_token)
        from mipha.constants import IMAGE_TOKEN_INDEX
        from mipha.conversation import SeparatorStyle, conv_templates

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

        vision_tower = self.model.get_vision_tower()
        mm_projector = self.model.get_model().mm_projector
        vision_raw = vision_tower(image_tensor)
        vision_projected = mm_projector(vision_raw)
        proj_feat = vision_projected.mean(dim=1).squeeze(0).float().cpu()

        q = question if "<image>" in question else "<image>\n" + question
        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
        input_ids = torch.as_tensor(input_ids).cuda().unsqueeze(0)
        input_token_len = input_ids.shape[1]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        out = self.model.generate(
            input_ids, images=image_tensor,
            do_sample=temperature > 0, temperature=temperature, top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
            stopping_criteria=[stopping_criteria],
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True, output_hidden_states=True,
        )
        # Mipha: 입력+생성 함께 반환 → 입력부 잘라냄
        gen_ids = out.sequences[:, input_token_len:]
        prediction = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        if prediction.endswith(stop_str):
            prediction = prediction[:-len(stop_str)].strip()

        prefill = out.hidden_states[0]
        feat = torch.stack([prefill[L][0, -1, :].float().cpu() for L in layers])
        return prediction, feat, proj_feat


# ── 벤치마크 파서 / 채점 (inf_mipha.py 와 동일) ───────────────
def parse_count(text, valid_range=(0, 2)):
    if not text:
        return None
    m = re.findall(r'\{\s*(-?\d+)\s*\}', text)
    if m:
        v = int(m[-1])
        return v if valid_range[0] <= v <= valid_range[1] else None
    m = re.findall(r'-?\d+', text)
    if m:
        v = int(m[-1])
        return v if valid_range[0] <= v <= valid_range[1] else None
    return None


def parse_grid(text, max_val=10):
    if not text:
        return None
    r = re.search(r'row\w*\s*=?\s*\{?\s*(\d+)', text, re.I)
    c = re.search(r'col\w*\s*=?\s*\{?\s*(\d+)', text, re.I)
    if r and c:
        rv, cv = int(r.group(1)), int(c.group(1))
        return (rv, cv) if (1 <= rv <= max_val and 1 <= cv <= max_val) else None

    nums = re.findall(r'\{\s*(\d+)\s*\}', text)
    if len(nums) >= 2:
        rv, cv = int(nums[0]), int(nums[1])
        return (rv, cv) if (1 <= rv <= max_val and 1 <= cv <= max_val) else None

    nums = re.findall(r'\d+', text)
    if len(nums) >= 2:
        rv, cv = int(nums[0]), int(nums[1])
        return (rv, cv) if (1 <= rv <= max_val and 1 <= cv <= max_val) else None

    return None


def parse_choice(text, valid="ab"):
    if not text:
        return None
    t = text.strip()
    m = re.search(r'[\(\[]?\s*([a-d])\s*[\)\].:]', t, re.I)
    if m and m.group(1).lower() in valid:
        return m.group(1).lower()
    m = re.match(r'\s*([a-d])\b', t, re.I)
    if m and m.group(1).lower() in valid:
        return m.group(1).lower()
    for m in re.finditer(r'\b([a-d])\b', t, re.I):
        if m.group(1).lower() in valid:
            return m.group(1).lower()
    return None


def task_kind(task_name):
    t = task_name.lower()
    if "line" in t:
        return "line_plot"
    if "count" in t or "grid" in t:
        return "counting_grid"
    if "mmvp" in t:
        return "mmvp"
    return "unknown"


def evaluate_one(ckpt_name, ds_name, records, dirs):
    kind = task_kind(ds_name)
    n = len(records)
    details = []
    summary = {"ckpt": ckpt_name, "task": ds_name, "kind": kind, "n": n}

    if kind == "line_plot":
        preds, gts, parsed = [], [], 0
        for r in records:
            p = parse_count(r["output"])
            g = str(r["expected_output"]).strip()
            parsed += p is not None
            preds.append(str(p) if p is not None else "<none>")
            gts.append(g)
            details.append({"id": r["id"], "pred": (str(p) if p is not None else None),
                            "gt": g, "correct": str(p) == g})
        labels = sorted(set(gts), key=lambda x: int(x))
        acc = float(np.mean([p == g for p, g in zip(preds, gts)])) if n else 0
        f1 = float(f1_score(gts, preds, labels=labels, average="macro", zero_division=0))
        summary.update({"parsed": parsed, "parse_ratio": parsed / n if n else 0,
                        "accuracy": acc, "f1": f1})
        print(f"  [{ckpt_name}/{ds_name}] (clf) acc={acc:.4f} F1={f1:.4f} parse={parsed}/{n}")

    elif kind == "counting_grid":
        parsed = exact = 0
        row_err, col_err, all_err = [], [], []
        for r in records:
            pr = parse_grid(r["output"])
            g = tuple(int(x) for x in str(r["expected_output"]).split(","))
            ok = pr is not None
            parsed += ok
            if ok:
                re_, ce_ = abs(pr[0] - g[0]), abs(pr[1] - g[1])
                row_err.append(re_); col_err.append(ce_); all_err += [re_, ce_]
                exact += (pr == g)
            details.append({"id": r["id"], "pred": pr, "gt": g,
                            "row_err": (abs(pr[0] - g[0]) if ok else None),
                            "col_err": (abs(pr[1] - g[1]) if ok else None)})
        summary.update({
            "parsed": parsed, "parse_ratio": parsed / n if n else 0,
            "mae": float(np.mean(all_err)) if all_err else float("nan"),
            "rows_mae": float(np.mean(row_err)) if row_err else float("nan"),
            "cols_mae": float(np.mean(col_err)) if col_err else float("nan"),
            "exact_acc": exact / n if n else 0,
        })
        print(f"  [{ckpt_name}/{ds_name}] (reg) MAE={summary['mae']:.4f} "
              f"exact={summary['exact_acc']:.4f} parse={parsed}/{n}")

    elif kind == "mmvp":
        preds, gts, parsed = [], [], 0
        for r in records:
            p = parse_choice(r["output"])
            g = str(r["expected_output"]).split(":")[0].strip().lower()
            parsed += p is not None
            preds.append(p if p is not None else "<none>")
            gts.append(g)
            details.append({"id": r["id"], "pred": p, "gt": g, "correct": p == g})
        labels = sorted(set(gts))
        acc = float(np.mean([p == g for p, g in zip(preds, gts)])) if n else 0
        f1 = float(f1_score(gts, preds, labels=labels, average="macro", zero_division=0))
        summary.update({"parsed": parsed, "parse_ratio": parsed / n if n else 0,
                        "accuracy": acc, "f1": f1})
        print(f"  [{ckpt_name}/{ds_name}] acc={acc:.4f} F1={f1:.4f} parse={parsed}/{n}")
    else:
        print(f"  [skip] {ckpt_name}/{ds_name}: unknown kind")
        return None

    os.makedirs(dirs["acc"], exist_ok=True)
    os.makedirs(dirs["parse"], exist_ok=True)
    json.dump(summary, open(os.path.join(dirs["acc"], f"{ckpt_name}__{ds_name}.json"), "w"), indent=2)
    json.dump(details, open(os.path.join(dirs["parse"], f"{ckpt_name}__{ds_name}.json"), "w"),
              indent=2, ensure_ascii=False)
    return summary


def process_dataset(loader, ckpt_name, ds_name, ds_path, temperature, dirs):
    samples = load_dataset(ds_path)
    records, feats, proj_feats = [], [], []
    predictions, answers, ids = [], [], []
    for s in tqdm(samples, desc=f"{ckpt_name}/{ds_name}", ncols=90):
        pred, feat, p_feat = loader.run_one(s["image_path"], s["question"], LAYERS, temperature)
        records.append(dict(id=s["id"], question=s["question"],
                            output=pred, expected_output=s["answer"]))
        feats.append(feat); proj_feats.append(p_feat)
        predictions.append(pred); answers.append(s["answer"]); ids.append(s["id"])

    feats = torch.stack(feats).half()
    proj_feats = torch.stack(proj_feats).half()

    pred_dir = os.path.join(dirs["pred"], ckpt_name)
    os.makedirs(pred_dir, exist_ok=True)
    with open(os.path.join(pred_dir, f"{ds_name}.json"), "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    emb_dir = os.path.join(dirs["emb"], ckpt_name)
    os.makedirs(emb_dir, exist_ok=True)
    torch.save({"feats": feats, "proj_feats": proj_feats, "predictions": predictions,
                "answers": answers, "ids": ids, "layers": LAYERS,
                "hidden_size": loader.hidden_size},
               os.path.join(emb_dir, f"{ds_name}.pt"))

    print(f"[saved] {ckpt_name}/{ds_name}  feats={tuple(feats.shape)} proj={tuple(proj_feats.shape)}")
    return evaluate_one(ckpt_name, ds_name, records, dirs)


def make_dirs(out):
    rdir = os.path.join(BASE, "result", out)
    return {"pred": os.path.join(rdir, "raw_responses"),
            "emb": os.path.join(rdir, "embeddings"),
            "acc": os.path.join(rdir, "accuracy"),
            "parse": os.path.join(rdir, "parsing")}


def save_summary(rows, dirs):
    os.makedirs(dirs["acc"], exist_ok=True)
    out = os.path.join(dirs["acc"], "_summary_all.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n전체 요약 → {out}")


def run(run_key, temperature):
    cfg = RUNS[run_key]
    dirs = make_dirs(cfg["out"])
    ckpts = build_ckpts(cfg["root"])
    print(f"[{run_key}] 처리할 ckpt {len(ckpts)}개: {list(ckpts)}")
    rows = []
    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n########## {run_key} / {ckpt_name} ##########")
        loader = MiphaInferExtract(ckpt_path)
        for ds_name in DATASETS:
            s = process_dataset(loader, ckpt_name, ds_name, DATASETS[ds_name], temperature, dirs)
            if s:
                rows.append(s)
        del loader
        torch.cuda.empty_cache()
    save_summary(rows, dirs)


def run_eval_only(run_key):
    dirs = make_dirs(RUNS[run_key]["out"])
    paths = sorted(glob.glob(os.path.join(dirs["pred"], "*", "*.json")))
    if not paths:
        print(f"raw_responses 없음. (dir={dirs['pred']})")
        return
    rows = []
    for p in paths:
        ckpt = os.path.basename(os.path.dirname(p))
        ds_name = os.path.basename(p).replace(".json", "")
        s = evaluate_one(ckpt, ds_name, json.load(open(p)), dirs)
        if s:
            rows.append(s)
    save_summary(rows, dirs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mipha", choices=list(RUNS))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--eval_only", action="store_true")
    a = ap.parse_args()
    if a.eval_only:
        run_eval_only(a.run)
    else:
        run(a.run, a.temperature)
