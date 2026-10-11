import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score

try:
    import wandb
except ImportError:
    wandb = None

EMB_ROOT = {
    "inf": "../extract_5_tasks/result/{model}/embedding",     # 폴더명: embedding(단수)
    "ben": "../extract_benchmark/result/{model}/embeddings",  # 폴더명: embeddings(복수)
}
MODEL  = "openflamingo"        # --model
SOURCE = "ben"            # --source (inf|ben)
SEEDS  = [53, 9, 144]     # 3 run: run i → seed SEEDS[i-1], 결과 → {prefix}_classification_{i}

EMB_DIR = EMB_ROOT[SOURCE].format(model=MODEL)
SEED    = SEEDS[0]        # run() 안에서 run 마다 교체
RES_DIR = f"./{MODEL}/{SOURCE}_classification_1"


def _res_dir(run_idx):
    suffix = "_linear" if PROBE_TYPE == "linear" else ""
    return f"./{MODEL}/{SOURCE}_classification{suffix}_{run_idx}"


def set_config(model, source):
    global MODEL, SOURCE, EMB_DIR
    MODEL, SOURCE = model, source
    EMB_DIR = EMB_ROOT[source].format(model=model)

PROBE_HIDDEN = 512
EPOCHS = 50
LR = 1e-3
BATCH = 64
N_FOLDS = 5
INNER_VAL = 0.15

WANDB_PROJECT = "probing"
USE_WANDB = True

SKIP_TASKS = {"S1_similarity", "counting_grid"}

# logmar: 정답이 6글자 문자열(예: 'MIZFPX') → 통짜 분류(180-class)는 무의미.
# 맨 앞글자만 학습/평가 (A~Z, 26-class)
FIRST_CHAR_TASKS = {"S0_logamr"}

PROBE_TYPE = "mlp"        # --probe mlp(2층 MLP512, 본 분석) | linear(은닉층 없는 softmax 선형 — 용량 대조군, 같은 학습 프로토콜)


class Probe(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=PROBE_HIDDEN):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:                                   # 선형 프로브: X → C 로짓 (multinomial logistic, 정규화는 inner-val early stopping 뿐 — MLP 와 동일)
            self.net = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.net(x)


def make_labels(answers):
    classes = sorted(set(answers))
    c2i = {c: i for i, c in enumerate(classes)}
    labels = np.array([c2i[a] for a in answers])
    return labels, len(classes)


def _split(X, y, test_size, seed):
    """stratify 시도 → 실패하면 일반 split."""
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=seed)


def _kfold_indices(X, y, n_splits, seed):
    """StratifiedKFold 시도 → 희귀 클래스 등으로 실패하면 일반 KFold."""
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(X, y))
    except ValueError:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(X))


def _fit_predict_fold(X_tr, y_tr, X_te, num_classes, hidden_size, dev):
    """fold 1개: train 안에서 inner val 떼서 best-epoch 선택 후 test fold 예측."""
    X_in, X_val, y_in, y_val = _split(X_tr, y_tr, INNER_VAL, SEED)

    scaler = StandardScaler()
    X_in = scaler.fit_transform(X_in)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    X_in = torch.tensor(X_in, dtype=torch.float32, device=dev)
    y_in = torch.tensor(y_in, dtype=torch.long, device=dev)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=dev)
    y_val = torch.tensor(y_val, dtype=torch.long, device=dev)
    X_te = torch.tensor(X_te, dtype=torch.float32, device=dev)

    probe = Probe(hidden_size, num_classes, hidden_dim=(None if PROBE_TYPE == "linear" else PROBE_HIDDEN)).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    n = len(X_in)
    for _ in range(EPOCHS):
        probe.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            crit(probe(X_in[idx]), y_in[idx]).backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            va = (probe(X_val).argmax(-1) == y_val).float().mean().item()
        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        return probe(X_te).argmax(-1).cpu().numpy()


def train_one_layer(X, y, num_classes, hidden_size, tag, run=None, layer=None):
    """K-fold OOF: 모든 샘플을 정확히 한 번씩 test에 넣어 예측을 모은 뒤 전체에서 평가."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    oof_pred = np.full(len(y), -1, dtype=np.int64)
    for tr_idx, te_idx in _kfold_indices(X, y, N_FOLDS, SEED):
        oof_pred[te_idx] = _fit_predict_fold(X[tr_idx], y[tr_idx], X[te_idx],
                                             num_classes, hidden_size, dev)

    acc = accuracy_score(y, oof_pred)
    f1 = f1_score(y, oof_pred, average="macro")
    print(f"    [{tag}] acc={acc:.4f}  macroF1={f1:.4f}  (OOF N={len(y)}, C={num_classes})")
    if run:
        wandb.log({"layer": layer, "test_acc": acc, "macro_f1": f1})
    return acc, f1


def process_pt(pt_path):
    ckpt_name = os.path.basename(os.path.dirname(pt_path))
    task_name = os.path.basename(pt_path).replace(".pt", "")
    name = f"{ckpt_name}__{task_name}"
    if task_name in SKIP_TASKS:
        print(f"[skip] {name} (회귀 task)")
        return None
    # 재개용: PROBE_SKIP_EXISTING=1 이면 이미 저장된 결과는 다시 계산하지 않음 (run_probing.sh 가 설정)
    if os.environ.get("PROBE_SKIP_EXISTING") == "1" and os.path.exists(os.path.join(RES_DIR, f"{name}.json")):
        print(f"[skip] {name} (이미 완료: {RES_DIR})")
        return None

    data = torch.load(pt_path, map_location="cpu")
    feats = data["feats"].float()
    answers = data["answers"]
    layers = data["layers"]
    hidden_size = data["hidden_size"]

    # logmar 등: 정답 문자열의 맨 앞글자만 분류 대상으로 (학습·평가 모두 첫 글자 기준)
    if task_name in FIRST_CHAR_TASKS:
        answers = [a[0] for a in answers]

    labels, num_classes = make_labels(answers)
    print(f"\n=== {name}  (N={len(answers)}, C={num_classes}) ===")

    run = None
    if USE_WANDB and wandb is not None:
        run = wandb.init(project=WANDB_PROJECT, name=f"{name}_s{SEED}" + ("_linear" if PROBE_TYPE == "linear" else ""),
                         config={"ckpt": ckpt_name, "task": task_name, "probe": PROBE_TYPE,
                                 "num_classes": num_classes, "seed": SEED}, reinit=True)

    result = {}
    for i, L in enumerate(layers):
        X = feats[:, i, :].numpy()
        acc, f1 = train_one_layer(X, labels, num_classes, X.shape[1],
                                  f"layer_{L}", run=run, layer=L)
        result[f"layer_{L}"] = {"acc": acc, "macro_f1": f1}

    if "proj_feats" in data:
        Xp = data["proj_feats"].float().numpy()
        acc, f1 = train_one_layer(Xp, labels, num_classes, Xp.shape[1],
                                  "proj", run=run, layer=-1)
        result["proj"] = {"acc": acc, "macro_f1": f1}

    if run:
        run.finish()

    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, f"{name}.json")
    json.dump({ckpt_name: {task_name: result}}, open(out, "w"), indent=2)
    print(f"    saved → {out}")
    return result


def run(ckpt=None, task=None):
    global SEED, RES_DIR
    pat = os.path.join(EMB_DIR, ckpt or "*", f"{task or '*'}.pt")
    pts = sorted(glob.glob(pat))
    if not pts:
        print(f"처리할 .pt 없음. (pat={pat})")
        return
    print(f"처리할 .pt: {len(pts)}개  (model={MODEL}, source={SOURCE}, probe={PROBE_TYPE}, runs={len(SEEDS)})")
    for run_idx, seed in enumerate(SEEDS, start=1):
        SEED = seed
        RES_DIR = _res_dir(run_idx)
        print(f"\n########## RUN {run_idx}/{len(SEEDS)}  seed={seed} → {RES_DIR} ##########")
        for pt in pts:
            process_pt(pt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--source", type=str, default=SOURCE, choices=list(EMB_ROOT))
    ap.add_argument("--probe", type=str, default=PROBE_TYPE, choices=["mlp", "linear"],
                    help="mlp=2층 MLP512(본 분석) / linear=은닉층 없는 선형 softmax(용량 대조군, GPU, 같은 분할·시드·early-stopping)")
    a = ap.parse_args()
    PROBE_TYPE = a.probe
    set_config(a.model, a.source)
    run(ckpt=a.ckpt, task=a.task)
