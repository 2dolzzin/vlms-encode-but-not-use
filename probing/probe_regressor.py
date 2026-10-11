# regressor.py
# color_similarity 처럼 정답이 연속값(0~1)인 task를 회귀 probe로 분석.
# result/vila_3b/embedding/{ckpt}/{task}.pt 를 읽어서 layer별 Ridge 회귀 → R²/MAE/...
#
# 분류(classifier.py)와 다른 점:
#   - 타겟이 float (answers="0.86" → 0.86)
#   - Ridge 선형회귀 + RidgeCV로 alpha 자동선택 (N=78 처럼 작은 데이터에 안정적)
#   - 단일 split 대신 5-fold OOF: 각 샘플을 정확히 한 번씩 test에 넣고
#     예측을 다 모아 R²/상관계수를 한 번에 계산 → 작은 N에서도 신뢰 가능
#   - 지표: R², MAE, RMSE, Pearson, Spearman
#
# 실행:
#   python regressor.py                       # 모든 ckpt × S1_similarity
#   python regressor.py --ckpt S0_end         # 특정 ckpt 하나
#   python regressor.py --task S1_similarity  # task 지정 (기본 S1_similarity)

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================
# 설정
# ============================================================
# source(inf=inference 5tasks / ben=benchmark) → 임베딩 루트.
# 출력 폴더 prefix 도 이걸로 결정: inf_regression_{run} / ben_regression_{run}
EMB_ROOT = {
    "inf": "../extract_5_tasks/result/{model}/embedding",     # 폴더명: embedding(단수)
    "ben": "../extract_benchmark/result/{model}/embeddings",  # 폴더명: embeddings(복수)
}
MODEL  = "openflamingo_revers"        # --model
SOURCE = "inf"            # --source (inf|ben)
# 3 run: run i → seed SEEDS[i-1] (classify_probe.py 와 동일 SEED 사용)
SEEDS  = [53, 9, 144]

EMB_DIR = EMB_ROOT[SOURCE].format(model=MODEL)
RES_DIR = f"./{MODEL}/{SOURCE}_regression_1"

def set_config(model, source):
    """모델·소스 설정 → 임베딩 경로 갱신. (RES_DIR/SEED 는 run() 에서 run별로 설정)"""
    global MODEL, SOURCE, EMB_DIR
    MODEL, SOURCE = model, source
    EMB_DIR = EMB_ROOT[source].format(model=model)

DEFAULT_TASK = "S1_similarity"
N_FOLDS = 5
ALPHAS  = np.logspace(-2, 4, 25)   # RidgeCV가 fold마다 이 중 best alpha 선택

SEED = SEEDS[0]   # run() 안에서 run 마다 교체

PROBE_TYPE = "ridge"   
# MLP probe 설정 (classifier.py와 동일 구조: 2560→512→ReLU→1)
MLP_HIDDEN = 512
MLP_EPOCHS = 50
MLP_LR     = 1e-3
MLP_BATCH  = 64
MLP_VAL    = 0.15      # fold train 안에서 best-model 선택용 val 비율

WANDB_PROJECT = "probing_llava"
USE_WANDB     = True


# ============================================================
# 타겟 만들기 (정답 문자열 → float)
# ============================================================
# regressor.py - make_targets 수정
def make_targets(answers, task_name=None):
    if task_name == "counting_grid":
        rows = np.array([float(a.split(',')[0]) for a in answers])
        cols = np.array([float(a.split(',')[1]) for a in answers])
        return rows, cols
    elif task_name == "S1_similarity": 
        return np.array([float(a) for a in answers], dtype=np.float64)

# ============================================================
# MLP probe (classifier.py와 동일: 2560→512→ReLU→1)
# ============================================================
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_predict_mlp(X_tr, y_tr, X_te):
    """fold 1개: train 안에서 val 떼서 best-model 선택 후 test 예측."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(SEED)

    # train 안에서 val 분리 (best model 선택용)
    n = len(X_tr)
    perm = rng.permutation(n)
    n_val = max(1, int(n * MLP_VAL))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    Xtr = torch.tensor(X_tr[tr_i], dtype=torch.float32, device=dev)
    ytr = torch.tensor(y_tr[tr_i], dtype=torch.float32, device=dev)
    Xva = torch.tensor(X_tr[val_i], dtype=torch.float32, device=dev)
    yva = torch.tensor(y_tr[val_i], dtype=torch.float32, device=dev)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=dev)

    probe = MLPProbe(X_tr.shape[1]).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=MLP_LR)
    crit = nn.MSELoss()

    best_val, best_state = float("inf"), None
    m = len(Xtr)
    for _ in range(MLP_EPOCHS):
        probe.train()
        order = torch.randperm(m, device=dev)
        for i in range(0, m, MLP_BATCH):
            idx = order[i:i + MLP_BATCH]
            opt.zero_grad()
            crit(probe(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            v = crit(probe(Xva), yva).item()
        if v < best_val:
            best_val = v
            best_state = {k: val.detach().clone() for k, val in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        return probe(Xte).cpu().numpy()


# ============================================================
# layer 1개 회귀 (5-fold OOF) — probe_type에 따라 Ridge / MLP
# ============================================================
def regress_one_layer(X, y, tag, run=None, layer=None, probe_type=None):
    probe_type = probe_type or PROBE_TYPE
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_pred = np.zeros_like(y)     # 각 샘플의 out-of-fold 예측
    alphas_used = []

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr       = y[tr_idx]

        # 정규화: train fit, test transform (layer 간 공정 비교)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        if probe_type == "mlp":
            oof_pred[te_idx] = _fit_predict_mlp(X_tr, y_tr, X_te)
        else:
            # RidgeCV: train 안에서 alpha 자동 선택 (efficient LOO)
            model = RidgeCV(alphas=ALPHAS)
            model.fit(X_tr, y_tr)
            oof_pred[te_idx] = model.predict(X_te)
            alphas_used.append(float(model.alpha_))

    # 78개 OOF 예측을 다 모아서 한 번에 평가
    r2   = r2_score(y, oof_pred)
    mae  = mean_absolute_error(y, oof_pred)
    rmse = np.sqrt(mean_squared_error(y, oof_pred))
    pear = pearsonr(y, oof_pred)[0]
    spear = spearmanr(y, oof_pred)[0]
    alpha_med = float(np.median(alphas_used)) if alphas_used else None

    extra = f"(alpha~{alpha_med:.3g})" if alpha_med is not None else "(mlp)"
    print(f"    [{tag}] R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"Pearson={pear:.4f}  Spearman={spear:.4f}  {extra}")
    if run:
        wandb.log({"layer": layer, "r2": r2, "mae": mae, "rmse": rmse,
                   "pearson": pear, "spearman": spear, "alpha": alpha_med})
    return {"r2": r2, "mae": mae, "rmse": rmse,
            "pearson": pear, "spearman": spear, "alpha": alpha_med}


# ============================================================
# .pt 1개 처리 (= ckpt × task 1개)
# ============================================================
def process_pt(pt_path, probe_type=None):
    probe_type = probe_type or PROBE_TYPE
    ckpt_name = os.path.basename(os.path.dirname(pt_path))
    task_name = os.path.basename(pt_path).replace(".pt", "")
    name = f"{ckpt_name}__{task_name}"
    # 재개용: PROBE_SKIP_EXISTING=1 이면 이미 저장된 결과는 다시 계산하지 않음 (run_probing.sh 가 설정)
    if os.environ.get("PROBE_SKIP_EXISTING") == "1" and os.path.exists(os.path.join(RES_DIR, probe_type, f"{name}.json")):
        print(f"[skip] {name} (이미 완료: {RES_DIR}/{probe_type})")
        return None

    data = torch.load(pt_path, map_location="cpu")
    feats   = data["feats"].float()
    answers = data["answers"]
    layers  = data["layers"]

    targets = make_targets(answers, task_name=task_name)

    run = None
    if USE_WANDB and wandb is not None:
        run = wandb.init(project=WANDB_PROJECT, name=f"{name}_s{SEED}",
                         config={"ckpt": ckpt_name, "task": task_name, "seed": SEED}, reinit=True)

    print(f"    probe={probe_type}")

    if task_name == "counting_grid":
        # rows, cols 각각 회귀 → 합쳐서 mae/rows_mae/cols_mae 산출 (ben_llava.py와 동일 형식)
        rows, cols = targets
        N = len(rows)
        print(f"\n=== {name}  (N={N}, rows: min={rows.min():.1f} max={rows.max():.1f}, "
              f"cols: min={cols.min():.1f} max={cols.max():.1f}) ===")

        result = {"_probe": probe_type}
        for i, L in enumerate(layers):
            X = feats[:, i, :].numpy()
            row_res = regress_one_layer(X, rows, f"layer_{L}_rows", run=run, layer=L, probe_type=probe_type)
            col_res = regress_one_layer(X, cols, f"layer_{L}_cols", run=run, layer=L, probe_type=probe_type)
            result[f"layer_{L}"] = {
                "rows_mae": row_res["mae"], "cols_mae": col_res["mae"],
                "mae": (row_res["mae"] + col_res["mae"]) / 2,
                "rows_r2": row_res["r2"], "cols_r2": col_res["r2"],
            }

        if "proj_feats" in data:
            Xp = data["proj_feats"].float().numpy()
            row_res = regress_one_layer(Xp, rows, "proj_rows", run=run, layer=-1, probe_type=probe_type)
            col_res = regress_one_layer(Xp, cols, "proj_cols", run=run, layer=-1, probe_type=probe_type)
            result["proj"] = {
                "rows_mae": row_res["mae"], "cols_mae": col_res["mae"],
                "mae": (row_res["mae"] + col_res["mae"]) / 2,
                "rows_r2": row_res["r2"], "cols_r2": col_res["r2"],
            }
    else:
        y = targets
        N = len(y)
        print(f"\n=== {name}  (N={N}, target: min={y.min():.2f} max={y.max():.2f} "
              f"mean={y.mean():.3f} std={y.std():.3f}) ===")
        base_mae = mean_absolute_error(y, np.full_like(y, y.mean()))
        print(f"    baseline(평균예측) MAE={base_mae:.4f}")

        result = {"_baseline_mae": base_mae, "_probe": probe_type}
        for i, L in enumerate(layers):
            X = feats[:, i, :].numpy()
            result[f"layer_{L}"] = regress_one_layer(X, y, f"layer_{L}", run=run, layer=L, probe_type=probe_type)

        if "proj_feats" in data:
            Xp = data["proj_feats"].float().numpy()
            result["proj"] = regress_one_layer(Xp, y, "proj", run=run, layer=-1, probe_type=probe_type)

    if run:
        run.finish()

    out_dir = os.path.join(RES_DIR, probe_type)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.json")
    json.dump({ckpt_name: {task_name: result}}, open(out, "w"), indent=2)
    print(f"    saved → {out}")
    return result


# ============================================================
# main
# ============================================================
def run(ckpt=None, task=DEFAULT_TASK, probe_type=None):
    global SEED, RES_DIR
    probe_type = probe_type or PROBE_TYPE
    if ckpt:
        pts = [os.path.join(EMB_DIR, ckpt, f"{task}.pt")]
    else:
        pts = sorted(glob.glob(os.path.join(EMB_DIR, "*", f"{task}.pt")))

    pts = [p for p in pts if os.path.exists(p)]
    if not pts:
        print(f"처리할 .pt 없음. (EMB_DIR={EMB_DIR}, task={task})")
        return

    print(f"처리할 .pt: {len(pts)}개  (model={MODEL}, source={SOURCE}, "
          f"task={task}, probe={probe_type}, runs={len(SEEDS)})")
    for run_idx, seed in enumerate(SEEDS, start=1):
        SEED = seed
        RES_DIR = f"./{MODEL}/{SOURCE}_regression_{run_idx}"
        print(f"\n########## RUN {run_idx}/{len(SEEDS)}  seed={seed} → {RES_DIR}/{probe_type} ##########")
        for pt in pts:
            process_pt(pt, probe_type=probe_type)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="특정 ckpt (예: S0_end)")
    ap.add_argument("--task", type=str, default=DEFAULT_TASK, help="task 이름 (기본 S1_similarity)")
    ap.add_argument("--probe", type=str, default=PROBE_TYPE, choices=["ridge", "mlp"],
                    help="probe 종류 (ridge=선형, mlp=2560→512→ReLU→1)")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--source", type=str, default=SOURCE, choices=list(EMB_ROOT),
                    help="inf=inference 5tasks / ben=benchmark (출력 prefix 도 결정)")
    a = ap.parse_args()
    set_config(a.model, a.source)
    run(ckpt=a.ckpt, task=a.task, probe_type=a.probe)
