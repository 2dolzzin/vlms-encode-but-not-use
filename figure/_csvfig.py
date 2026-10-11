"""csv_out/ 의 CSV 를 단일 소스로 figure 를 그리는 공용 유틸.
가로=시점(체크포인트, 학습진행순), 세로=test. metric=='mae' 인 행만 표시용 1-mae 로 변환.
"""
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 저장소 루트 (figure/ 의 상위)
CSVDIR = os.path.join(BASE, "csv_out")
FIGDIR = os.path.join(BASE, "figure")

# 실험별 출력 디렉토리 (figure/<EXP_DIR>/...). out_stem 앞에 붙여서 쓴다.
EXP_DIR = {
    "exp1": "exp1_curriculum_inference",   # Exp1 커리큘럼 5-task 추론
    "exp2": "exp2_curriculum_probing",     # Exp2 커리큘럼 프로빙
    "exp3": "exp3_benchmark_inference",    # Exp3 blind 벤치마크 + line_plot + MMVP 추론
    "exp4": "exp4_benchmark_probing",      # Exp4 벤치마크 프로빙
}


def save_fig(fig, out_stem, dpi):
    """figure/<out_stem>.{png,pdf} 저장 (하위 디렉토리 자동 생성)."""
    path = os.path.join(FIGDIR, out_stem)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=dpi)

# VILA-8B 는 랜덤 순서 런이 없어 figure 에서 제외(2026-10). Qwen2.5-VL-7B 추가.
MODELS = ["llava", "mipha", "vila_3b", "openflamingo", "qwen25vl_7b"]
MODEL_COLORS = {"llava": "#1f77b4", "mipha": "#2ca02c", "vila_3b": "#ff7f0e",
                "openflamingo": "#9467bd", "qwen25vl_7b": "#d62728"}
MODEL_MARKERS = {"llava": "o", "mipha": "s", "vila_3b": "^", "openflamingo": "v", "qwen25vl_7b": "D"}
MODEL_LABELS = {"llava": "LLaVA-1.5", "mipha": "Mipha-3B", "vila_3b": "VILA-3B",
                "openflamingo": "OpenFlamingo", "qwen25vl_7b": "Qwen2.5-VL"}

# ── 학습 순서(variant) ────────────────────────────────────────
# csv_out/{inf,probing}_<name>_<variant>.csv 의 접미사. 랜덤 순서는 order_<tag>.
VARIANTS = ["forward", "reverse", "order_14203", "order_20431", "order_31024"]
VARIANT_COLORS = {"forward": "#1f77b4", "reverse": "#d62728", "order_14203": "#2ca02c",
                  "order_20431": "#ff7f0e", "order_31024": "#9467bd"}
VARIANT_MARKERS = {"forward": "o", "reverse": "s", "order_14203": "^", "order_20431": "v", "order_31024": "D"}

# inference 행에서 task 별로 쓸 metric. mmvp 는 pair_accuracy 만 사용(질문단위 accuracy 행은 버림).
INF_METRIC = {"mmvp": "pair_accuracy"}

# probing layer 는 '절대 인덱스'가 아니라 '깊이 순번(L0~L4)+proj' 로 정규화.
# 모델마다 층수가 달라(7B계열 32층 → 0/8/16/24/32, OpenFlamingo 24층 → 0/6/12/18/24)
# 각 모델이 샘플한 5개 깊이를 얕→깊 순서대로 L0~L4 로 매핑 → 같은 순번=같은 상대깊이.
DEPTH_COLORS = {"L0": "#bbbbbb", "L1": "#9ecae1", "L2": "#4292c6",
                "L3": "#2171b5", "L4": "#084594", "proj": "#e07b39"}
DEPTH_MARKERS = {"L0": "o", "L1": "o", "L2": "s", "L3": "^", "L4": "v", "proj": "D"}
DEPTH_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "proj": 5}


def depth_key(layer, sorted_layers):
    """raw layer 이름 → 깊이 순번 라벨(L0~L4)/proj. sorted_layers=모델의 얕→깊 정렬된 수치 레이어."""
    if layer == "proj":
        return "proj"
    n = int(layer.split("_")[1])
    return f"L{sorted_layers.index(n)}"


INF_COLOR = "black"
CHANCE_COLOR = "#999999"
STAGE_SHADE = "#b8d9f0"   # 학습 stage 음영(하늘색)

# ── 폰트 크기(전부 크게: 기본 38) ──────────────────────────────
FS_TITLE = 38     # 열 제목(task)
FS_LABEL = 38     # 축 라벨
FS_TICK = 38      # 눈금 라벨
FS_TAG = 38       # 행 바깥 (a)/(b) 태그
FS_LEGEND = 38    # 범례
FS_YTICK = 32     # y축 숫자(0.25/0.50/...) 는 조금 작게


def stage_span(ckcols, task):
    """task(예: 'S2_ishihara')가 학습되는 mid~end 구간의 x범위. curriculum(S\\d) 만."""
    stage = task.split("_")[0]
    if not (len(stage) == 2 and stage[0] == "S" and stage[1].isdigit()):
        return None
    idxs = [i for i, c in enumerate(ckcols)
            if c.split("_")[0] == stage and (c.endswith("_mid") or c.endswith("_end"))]
    return (min(idxs) - 0.5, max(idxs) + 0.5) if idxs else None


def _fv(s):
    return float("nan") if s == "" else float(s)


def load(path):
    """return (ckpt_cols, rows) ; rows = list of dict(id-fields..., _s={ckpt:val})."""
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        bi = cols.index("Base")
        idcols, ckcols = cols[:bi], cols[bi:]
        rows = []
        for r in rd:
            d = {c: r[c] for c in idcols}
            # inference 파일(layer 열 없음): 같은 task 에 metric 이 여러 개면 INF_METRIC 의 것만 사용
            if "layer" not in idcols and d.get("task") in INF_METRIC and d.get("metric") != INF_METRIC[d["task"]]:
                continue
            d["_s"] = {c: _fv(r[c]) for c in ckcols}
            rows.append(d)
    return ckcols, rows


def variant_label(variant, ckcols=None, short=False):
    """라벨. forward/reverse 는 그대로, 랜덤은 'Random (S2→S0→S4→S3→S1)' (short=True 면 'S2→S0→S4→S3→S1').
    short 는 블록 태그용(길면 옆 패널 y축 라벨과 겹침)."""
    if variant in ("forward", "reverse"):
        return variant.capitalize()
    if ckcols:
        seq = "→".join(c.split("_")[0] for c in ckcols[1::2])
        return seq if short else f"Random ({seq})"
    return "Random " + variant.split("_", 1)[1]


def load_variant(kind, name, variant):
    """kind='inf'|'probing'. 없으면 (None, [])."""
    path = os.path.join(CSVDIR, f"{kind}_{name}_{variant}.csv")
    if not os.path.exists(path):
        print(f"[warn] {os.path.basename(path)} 없음 → 건너뜀")
        return None, []
    return load(path)


def tick_info(ckcols):
    stages = [c.split("_")[0] for c in ckcols[1::2]]
    labels = ["Base"] + stages
    pos = [0] + [1.5 + 2 * i for i in range(len(stages))]
    return pos, labels


def disp(series, ckcols, metric):
    """표시용 값 리스트. mae → 1-mae, 그 외 raw."""
    ys = [series[c] for c in ckcols]
    if metric == "mae":
        ys = [1 - v if not math.isnan(v) else np.nan for v in ys]
    return ys


def ylabel_for(metric):
    return "1 − MAE" if metric == "mae" else "Accuracy"


def metric_abbr(metric):
    """y축 라벨 괄호 안 약어. accuracy→Acc, mae→1−MAE."""
    return "1−MAE" if metric == "mae" else "Acc"


# ══════════════════════════════════════════════════════════════
# INFERENCE figure : 2행(fwd/rev) × task열, 선=모델
# ══════════════════════════════════════════════════════════════
def plot_inf(name, tasks, chance, out_stem, panel_w=3.0, shade_stage=False):
    """tasks: [(task_key, label)] ; chance: {task_key: val or None}.
    shade_stage=True 면 curriculum task 의 학습구간(mid~end)에 음영."""
    fwd_ck, fwd_rows = load(os.path.join(CSVDIR, f"inf_{name}_forward.csv"))
    rev_ck, rev_rows = load(os.path.join(CSVDIR, f"inf_{name}_reverse.csv"))
    DIRS = [("Forward", fwd_ck, fwd_rows), ("Reverse", rev_ck, rev_rows)]

    ncol = len(tasks)
    fig, axes = plt.subplots(2, ncol, figsize=(panel_w * ncol, 12), squeeze=False)
    fig.subplots_adjust(hspace=0.28, wspace=0.55, left=0.08)

    any_shade = False
    for ri, (dlabel, ckcols, rows) in enumerate(DIRS):
        pos, ticklab = tick_info(ckcols)
        by = {(r["task"], r["model"]): r for r in rows}
        metric_of = {r["task"]: r["metric"] for r in rows}
        for ci, (task, tlabel) in enumerate(tasks):
            ax = axes[ri][ci]
            metric = metric_of.get(task, "accuracy")
            if shade_stage:
                span = stage_span(ckcols, task)
                if span:
                    ax.axvspan(span[0], span[1], alpha=0.3, color=STAGE_SHADE, zorder=0)
                    any_shade = True
            for model in MODELS:
                r = by.get((task, model))
                if not r:
                    continue
                ax.plot(range(len(ckcols)), disp(r["_s"], ckcols, metric),
                        color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                        markersize=4, linewidth=1.7, label=MODEL_LABELS[model])
            ch = chance.get(task)
            if ch is not None:
                ax.axhline(ch, color=CHANCE_COLOR, ls="--", lw=1, alpha=0.7)
            ax.set_xticks(pos); ax.set_xticklabels(ticklab, fontsize=FS_TICK)
            ax.set_xlim(-0.5, len(ckcols) - 0.5)
            ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(axis="y", labelsize=FS_TICK, labelleft=True)   # 모든 패널에 y눈금
            ax.set_ylabel(ylabel_for(metric), fontsize=FS_LABEL, labelpad=6)  # 패널마다 metric 명시
            ax.grid(axis="y", alpha=0.3, ls=":")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if ri == 0:
                ax.set_title(tlabel, fontsize=FS_TITLE, fontweight="bold", pad=6)

    ROW_TAGS = ["(a) Forward", "(b) Reverse"]
    for ri, tag in enumerate(ROW_TAGS):
        p = axes[ri][0].get_position()
        fig.text(p.x0 - 0.065, (p.y0 + p.y1) / 2, tag, rotation=90,
                 va="center", ha="center", fontsize=FS_TAG, fontweight="bold")

    handles = [mlines.Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                             markersize=8, lw=2, label=MODEL_LABELS[m]) for m in MODELS]
    handles.append(mlines.Line2D([0], [0], color=CHANCE_COLOR, ls="--", lw=1.5, label="Chance"))
    if any_shade:
        handles.append(mpatches.Patch(color=STAGE_SHADE, alpha=0.5, label="Training stage"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=FS_LEGEND, bbox_to_anchor=(0.5, -0.03), frameon=True, edgecolor="#cccccc")

    save_fig(fig, out_stem, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_stem}.png / .pdf   ({ncol} tasks)")


# ══════════════════════════════════════════════════════════════
# INFERENCE figure (래핑판) : task 를 wrap개씩 줄바꿈.
#   Forward 블록(위) / Reverse 블록(아래). (a)/(b)는 각 블록 왼쪽위 1회.
#   패널 title 없이 y축 라벨에 'Task (Acc)' 표기.
# ══════════════════════════════════════════════════════════════
def plot_inf_wrap(name, tasks, chance, out_stem, wrap=3, panel_w=8.5, panel_h=7.2,
                  shade_stage=False, legend_y=-0.045, tag_dx=-0.1, tag_dy=0.02,
                  block_gap=0.03, variants=("forward", "reverse"), tag_labels=False):
    """variants: 블록 순서(위→아래). tag_labels=True 면 '(c) Random (S2→…)' 식으로 라벨 병기."""
    DIRS = []
    for vi, v in enumerate(variants):
        ck, rows = load_variant("inf", name, v)
        if ck is None:
            continue
        tag = f"({chr(97 + len(DIRS))})" + (f" {variant_label(v, ck, short=True)}" if tag_labels else "")
        DIRS.append((tag, ck, rows))

    ncol = len(tasks)
    nblk = math.ceil(ncol / wrap)      # 방향당 행 수
    nrow = nblk * len(DIRS)
    fig, axes = plt.subplots(nrow, wrap, figsize=(panel_w * wrap, panel_h * nrow),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.60, wspace=0.58, left=0.07, right=0.99,
                        top=0.93, bottom=0.06)

    any_shade = False
    for di, (dtag, ckcols, rows) in enumerate(DIRS):
        pos, ticklab = tick_info(ckcols)
        by = {(r["task"], r["model"]): r for r in rows}
        metric_of = {r["task"]: r["metric"] for r in rows}
        base = di * nblk
        for ti, (task, tlabel) in enumerate(tasks):
            ax = axes[base + ti // wrap][ti % wrap]
            metric = metric_of.get(task, "accuracy")
            if shade_stage:
                span = stage_span(ckcols, task)
                if span:
                    ax.axvspan(span[0], span[1], alpha=0.3, color=STAGE_SHADE, zorder=0)
                    any_shade = True
            for model in MODELS:
                r = by.get((task, model))
                if not r:
                    continue
                ax.plot(range(len(ckcols)), disp(r["_s"], ckcols, metric),
                        color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                        markersize=5, linewidth=2.0, label=MODEL_LABELS[model])
            ch = chance.get(task)
            if ch is not None:
                ax.axhline(ch, color=CHANCE_COLOR, ls="--", lw=1, alpha=0.7)
            ax.set_xticks(pos)
            ax.set_xticklabels(ticklab, fontsize=FS_TICK - 4, fontweight="bold")
            ax.get_xticklabels()[0].set_ha("right")   # 'Base' 를 왼쪽으로 밀어 0.00 자리 피함
            ax.set_xlim(-0.5, len(ckcols) - 0.5)
            ax.set_ylim(0, 1.05); ax.set_yticks([0.25, 0.5, 0.75, 1.0])  # 0.00 라벨 제거(Base와 겹침)
            ax.tick_params(axis="y", labelsize=FS_YTICK, labelleft=True)
            plt.setp(ax.get_yticklabels(), fontweight="bold")
            # title 대신 y축 라벨에 task + metric 약어
            ax.set_ylabel(f"{tlabel} ({metric_abbr(metric)})", fontsize=FS_LABEL,
                          fontweight="bold", labelpad=8)
            ax.grid(axis="y", alpha=0.3, ls=":")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        # 남는 빈 셀 숨김
        for idx in range(ncol, nblk * wrap):
            axes[base + idx // wrap][idx % wrap].axis("off")

    # 두 번째 블록부터 아래로 내려 블록 간 간격 벌림(블록 k 는 k*block_gap 만큼)
    if block_gap:
        for row in range(nblk, nrow):
            for c in range(wrap):
                bb = axes[row][c].get_position()
                axes[row][c].set_position([bb.x0, bb.y0 - block_gap * (row // nblk), bb.width, bb.height])

    # (a)/(b) 는 각 블록 맨 위 왼쪽에 한 번만(기울이지 않음)
    for di, (dtag, ckcols, rows) in enumerate(DIRS):
        p = axes[di * nblk][0].get_position()
        fig.text(p.x0 + tag_dx, p.y1 + tag_dy, dtag, ha="left", va="bottom",
                 fontsize=FS_TAG + 8, fontweight="bold")

    handles = [mlines.Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                             markersize=8, lw=2, label=MODEL_LABELS[m]) for m in MODELS]
    handles.append(mlines.Line2D([0], [0], color=CHANCE_COLOR, ls="--", lw=1.5, label="Chance"))
    if any_shade:
        handles.append(mpatches.Patch(color=STAGE_SHADE, alpha=0.5, label="Training stage"))
    ncol_leg = math.ceil(len(handles) / 2)   # 범례 2줄
    fig.legend(handles=handles, loc="lower center", ncol=ncol_leg,
               prop={"size": FS_LEGEND, "weight": "bold"},
               bbox_to_anchor=(0.5, legend_y), frameon=True, edgecolor="#cccccc")

    save_fig(fig, out_stem, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_stem}.png / .pdf   ({ncol} tasks, wrap={wrap})")


# ══════════════════════════════════════════════════════════════
# PROBING figure : (모델×2)행 × task열, 선=layer + inference(별)
# ══════════════════════════════════════════════════════════════
def plot_probing(name, tasks, chance, out_stem, panel_w=2.3, panel_h=3.6, shade_stage=False):
    """tasks: [(task_key, label)]. inference 오버레이는 inf_{name}_*.csv 에서.
    각 열 제목에 measurement(Accuracy / 1−MAE) 표기. shade_stage=True 면 학습구간 음영."""
    p_fwd_ck, p_fwd = load(os.path.join(CSVDIR, f"probing_{name}_forward.csv"))
    p_rev_ck, p_rev = load(os.path.join(CSVDIR, f"probing_{name}_reverse.csv"))
    i_fwd_ck, i_fwd = load(os.path.join(CSVDIR, f"inf_{name}_forward.csv"))
    i_rev_ck, i_rev = load(os.path.join(CSVDIR, f"inf_{name}_reverse.csv"))

    # task -> metric (probe clf=acc, S1=mae). 열 제목 measurement 표기에 사용.
    task_metric = {}
    for r in p_fwd:
        task_metric.setdefault(r["task"], r["metric"])

    # 모델별 수치 레이어를 얕→깊 정렬(깊이 순번 L0~L4 매핑에 사용).
    model_layers = {}
    for r in p_fwd:
        if r["layer"].startswith("layer_"):
            model_layers.setdefault(r["model"], set()).add(int(r["layer"].split("_")[1]))
    model_layers = {m: sorted(s) for m, s in model_layers.items()}

    def index_probe(rows):
        d = {}
        for r in rows:
            d.setdefault((r["model"], r["task"]), []).append(r)
        return d

    def index_inf(rows):
        return {(r["model"], r["task"]): r for r in rows}

    DIRS = [("Fwd", p_fwd_ck, index_probe(p_fwd), index_inf(i_fwd)),
            ("Rev", p_rev_ck, index_probe(p_rev), index_inf(i_rev))]

    ncol = len(tasks)
    nrow = len(MODELS) * 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow), squeeze=False)
    fig.subplots_adjust(hspace=0.68, wspace=0.42, left=0.1)

    seen_depths = set()
    any_shade = False
    for mi, model in enumerate(MODELS):
        slayers = model_layers.get(model, [0, 8, 16, 24, 32])
        for di, (dlabel, ckcols, pidx, iidx) in enumerate(DIRS):
            row = mi * 2 + di
            pos, ticklab = tick_info(ckcols)
            for ci, (task, tlabel) in enumerate(tasks):
                ax = axes[row][ci]
                if shade_stage:
                    span = stage_span(ckcols, task)
                    if span:
                        ax.axvspan(span[0], span[1], alpha=0.3, color=STAGE_SHADE, zorder=0)
                        any_shade = True
                # probe lines (깊이 순번 별: L0~L4 + proj)
                for r in sorted(pidx.get((model, task), []),
                                key=lambda x: DEPTH_ORDER.get(depth_key(x["layer"], slayers), 9)):
                    dk = depth_key(r["layer"], slayers); seen_depths.add(dk)
                    ax.plot(range(len(ckcols)), disp(r["_s"], ckcols, r["metric"]),
                            color=DEPTH_COLORS.get(dk, "#777777"),
                            marker=DEPTH_MARKERS.get(dk, "o"),
                            markersize=2.5, lw=1.0, zorder=3)
                # inference star
                ir = iidx.get((model, task))
                if ir:
                    ax.plot(range(len(ckcols)), disp(ir["_s"], ckcols, ir["metric"]),
                            color=INF_COLOR, marker="*", markersize=5, lw=1.2, zorder=5)
                ch = chance.get(task)
                if ch is not None:
                    ax.axhline(ch, color=CHANCE_COLOR, ls=":", lw=0.8, zorder=2)
                ax.set_xticks(pos[::1]); ax.set_xticklabels(ticklab, fontsize=FS_TICK)
                ax.set_xlim(-0.5, len(ckcols) - 0.5)
                ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1.0])
                ax.tick_params(axis="y", labelsize=FS_TICK)
                ax.grid(axis="y", alpha=0.15, ls=":")
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                if row == 0:
                    # 열 제목에 task + measurement(y축 의미) 명시
                    ax.set_title(f"{tlabel}\n[{ylabel_for(task_metric.get(task, 'acc'))}]",
                                 fontsize=FS_TITLE, fontweight="bold", pad=5)
                if ci == 0:
                    ax.set_ylabel(ylabel_for(task_metric.get(task, "acc")), fontsize=FS_LABEL, labelpad=4)

    # 모델(a=Fwd / b=Rev) 라벨은 각 행 왼쪽 바깥에
    for mi, model in enumerate(MODELS):
        for di in range(2):
            row = mi * 2 + di
            p = axes[row][0].get_position()
            fig.text(p.x0 - 0.055, (p.y0 + p.y1) / 2,
                     f"({'a' if di == 0 else 'b'}) {MODEL_LABELS[model]}",
                     rotation=90, va="center", ha="center", fontsize=FS_TAG, fontweight="bold")

    # legend : 실제 등장한 깊이 순번(L0~L4, proj)
    order = sorted(seen_depths, key=lambda d: DEPTH_ORDER.get(d, 9))
    handles = [mlines.Line2D([0], [0], color=DEPTH_COLORS.get(d, "#777777"),
                             marker=DEPTH_MARKERS.get(d, "o"), markersize=7, lw=1.8,
                             label=d) for d in order]
    handles.append(mlines.Line2D([0], [0], color=INF_COLOR, marker="*", markersize=9, lw=2, label="Inference"))
    handles.append(mlines.Line2D([0], [0], color=CHANCE_COLOR, ls=":", lw=1.3, label="Chance"))
    if any_shade:
        handles.append(mpatches.Patch(color=STAGE_SHADE, alpha=0.5, label="Training stage"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 10),
               fontsize=FS_LEGEND, bbox_to_anchor=(0.5, 0.045), frameon=True, edgecolor="#cccccc")

    save_fig(fig, out_stem, dpi=130)
    plt.close(fig)
    print(f"Saved -> {out_stem}.png / .pdf   ({ncol} tasks × {nrow} rows)")


def _lk(layer):
    if layer == "proj":
        return (1, 0)
    return (0, int(layer.split("_")[1]))


def load_probe_dirs(name, variants, probe="mlp"):
    """probing+inf CSV 를 variant 별로 로드.
    probe='mlp'    → probing_<name>_<variant>.csv (MLP512 분류기 + S1 ridge)
    probe='linear' → probing_<name>_linear_<variant>.csv (GPU 선형 분류기) + S1 ridge 행은 MLP 파일에서 가져옴
    return DIRS=[(letter, vlabel, ckcols, probe_idx, inf_idx)], task_metric, model_layers"""
    assert probe in ("mlp", "linear"), probe
    DIRS, task_metric, model_layers = [], {}, {}
    for v in variants:
        if probe == "mlp":
            p_ck, p_rows = load_variant("probing", name, v)
        else:
            p_ck, p_rows = load_variant("probing", name + "_linear", v)
            if p_ck is not None:   # ridge(S1 회귀, metric=mae) 는 분류기와 무관 → MLP 파일에서 보충
                m_ck, m_rows = load_variant("probing", name, v)
                p_rows = p_rows + [r for r in m_rows if r["metric"] != "acc"]
        if p_ck is None:
            continue
        i_ck, i_rows = load_variant("inf", name, v)
        for r in p_rows:
            task_metric.setdefault(r["task"], r["metric"])
            if r["layer"].startswith("layer_"):
                model_layers.setdefault(r["model"], set()).add(int(r["layer"].split("_")[1]))
        pidx = {}
        for r in p_rows:
            pidx.setdefault((r["model"], r["task"]), []).append(r)
        iidx = {(r["model"], r["task"]): r for r in i_rows}
        DIRS.append((chr(97 + len(DIRS)), variant_label(v, p_ck, short=True), p_ck, pidx, iidx))
    model_layers = {m: sorted(s) for m, s in model_layers.items()}
    return DIRS, task_metric, model_layers


def layer_legend_groups(models, model_layers):
    """서로 다른 레이어 번호 집합을 모델 등장 순서대로 → ([layers...], 'LLaVA, Mipha, VILA / OpenFlamingo / Qwen')"""
    groups, names = [], []
    for m in models:
        ls = model_layers.get(m)
        if not ls:
            continue
        if ls in groups:
            names[groups.index(ls)].append(MODEL_LABELS[m])
        else:
            groups.append(ls); names.append([MODEL_LABELS[m]])
    return groups, " / ".join(", ".join(n) for n in names)


# ══════════════════════════════════════════════════════════════
# INFERENCE figure (순서 비교판) : x축 = 스테이지 이름(S0~S4) 고정, 선 = 학습 순서(variant).
#   패널 = task 행 × 모델 열. 같은 스테이지를 '학습한 직후' 점수가 순서에 따라 얼마나 다른지 비교.
#   학습 순서가 달라도 열(S*_mid/S*_end)은 스테이지 이름이므로 이름 기준으로 재배열.
# ══════════════════════════════════════════════════════════════
STAGE_COLS = ["Base"] + [f"S{k}_{p}" for k in range(5) for p in ("mid", "end")]


def plot_inf_bystage(name, tasks, chance, out_stem, variants=VARIANTS, models=None,
                     panel_w=7.0, panel_h=6.5, shade_stage=False, legend_y=-0.01):
    models = models if models is not None else MODELS
    DATA = []   # (variant, ckcols, {(task,model): row})
    for v in variants:
        ck, rows = load_variant("inf", name, v)
        if ck is None:
            continue
        DATA.append((v, ck, {(r["task"], r["model"]): r for r in rows}))
    metric_of = {}
    for _, _, by in DATA:
        for (task, _m), r in by.items():
            metric_of.setdefault(task, r["metric"])

    nrow, ncol = len(tasks), len(models)
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.45, left=0.06, right=0.99, top=0.96, bottom=0.04)
    pos, ticklab = tick_info(STAGE_COLS)
    any_shade = False
    for ti, (task, tlabel) in enumerate(tasks):
        metric = metric_of.get(task, "accuracy")
        for mi, model in enumerate(models):
            ax = axes[ti][mi]
            if shade_stage:
                span = stage_span(STAGE_COLS, task)
                if span:
                    ax.axvspan(span[0], span[1], alpha=0.3, color=STAGE_SHADE, zorder=0); any_shade = True
            for v, ck, by in DATA:
                r = by.get((task, model))
                if not r:
                    continue
                ax.plot(range(len(STAGE_COLS)), disp(r["_s"], STAGE_COLS, metric),
                        color=VARIANT_COLORS[v], marker=VARIANT_MARKERS[v], markersize=5, lw=2.0)
            ch = chance.get(task)
            if ch is not None:
                ax.axhline(ch, color=CHANCE_COLOR, ls="--", lw=1, alpha=0.7)
            ax.set_xticks(pos); ax.set_xticklabels(ticklab, fontsize=FS_TICK - 6, fontweight="bold")
            ax.get_xticklabels()[0].set_ha("right")
            ax.set_xlim(-0.5, len(STAGE_COLS) - 0.5)
            ax.set_ylim(0, 1.05); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.tick_params(axis="y", labelsize=FS_YTICK, labelleft=True)
            plt.setp(ax.get_yticklabels(), fontweight="bold")
            ax.grid(axis="y", alpha=0.3, ls=":")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if ti == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=FS_TITLE, fontweight="bold", pad=10)
            if mi == 0:
                ax.set_ylabel(f"{tlabel} ({metric_abbr(metric)})", fontsize=FS_LABEL, fontweight="bold", labelpad=8)

    handles = [mlines.Line2D([0], [0], color=VARIANT_COLORS[v], marker=VARIANT_MARKERS[v], markersize=9, lw=2.5,
                             label=variant_label(v, ck)) for v, ck, _ in DATA]
    handles.append(mlines.Line2D([0], [0], color=CHANCE_COLOR, ls="--", lw=1.5, label="Chance"))
    if any_shade:
        handles.append(mpatches.Patch(color=STAGE_SHADE, alpha=0.5, label="Training stage"))
    fig.legend(handles=handles, loc="upper center", ncol=math.ceil(len(handles) / 2),
               prop={"size": FS_LEGEND, "weight": "bold"}, bbox_to_anchor=(0.5, legend_y),
               frameon=True, edgecolor="#cccccc")
    save_fig(fig, out_stem, dpi=110)
    plt.close(fig)
    print(f"Saved -> {out_stem}.png / .pdf   ({nrow} tasks × {ncol} models, {len(DATA)} orders)")


# ══════════════════════════════════════════════════════════════
# PROBING figure (래핑판) : (모델×방향)이 한 블록. task 를 wrap개씩 줄바꿈.
#   각 블록 좌상단에 '(a) LLaVA-1.5' 식 태그(a=Fwd/b=Rev), 패널 title 없이
#   y축 라벨에 'Task (Acc)'. 선=깊이(L0~L4+proj), inference=별.
# ══════════════════════════════════════════════════════════════
def plot_probing_wrap(name, tasks, chance, out_stem, wrap=5, panel_w=7.0, panel_h=7.6,
                      shade_stage=False, legend_y=-0.02, tag_dx=-0.1, tag_dy=0.012, hspace=0.30, gap_ratio=0.45,
                      layer_legend=False, models=None, show_legend=True,
                      variants=("forward", "reverse"), tag_labels=False, probe="mlp"):
    """variants: 모델마다 이 순서로 블록. tag_labels=True 면 태그에 variant 라벨 병기. probe='mlp'|'linear'."""
    DIRS, task_metric, model_layers = load_probe_dirs(name, variants, probe)

    # 블록 = 모델마다 variants 순서 (기본 (a)Fwd, (b)Rev)
    models = models if models is not None else MODELS
    BLOCKS = []
    for model in models:
        for letter, vlabel, ckcols, pidx, iidx in DIRS:
            tag = f"({letter}) {MODEL_LABELS[model]}" + (f" — {vlabel}" if tag_labels else "")
            BLOCKS.append((tag, model, ckcols, pidx, iidx))

    ncol = len(tasks)
    nbr = math.ceil(ncol / wrap)      # 블록당 행 수
    B = len(BLOCKS)
    # 블록 사이에 빈 spacer 행을 끼워 '블록 간 간격 > 블록 내 행 간격' 확보
    n_spacer = B - 1
    total_rows = B * nbr + n_spacer
    height_ratios = []
    for b in range(B):
        height_ratios += [1.0] * nbr
        if b < B - 1:
            height_ratios.append(gap_ratio)
    fig_h = panel_h * (B * nbr + gap_ratio * n_spacer)
    fig = plt.figure(figsize=(panel_w * wrap, fig_h))
    gs = fig.add_gridspec(total_rows, wrap, height_ratios=height_ratios, hspace=hspace, wspace=0.5, left=0.06, right=0.99,
                          top=0.985, bottom=0.02)
    axmap = {}          # (bi, r, c) -> ax
    block_top_ax = {}   # bi -> 블록 첫행 첫열 ax (태그 기준점)
    grow = 0
    for bi in range(B):
        for r in range(nbr):
            for c in range(wrap):
                ax = fig.add_subplot(gs[grow, c])
                axmap[(bi, r, c)] = ax
                if r == 0 and c == 0:
                    block_top_ax[bi] = ax
            grow += 1
        if bi < B - 1:
            grow += 1   # spacer 건너뜀

    seen_depths = set()
    any_shade = False
    for bi, (tag, model, ckcols, pidx, iidx) in enumerate(BLOCKS):
        slayers = model_layers.get(model, [0, 8, 16, 24, 32])
        pos, ticklab = tick_info(ckcols)
        for ti, (task, tlabel) in enumerate(tasks):
            ax = axmap[(bi, ti // wrap, ti % wrap)]
            metric = task_metric.get(task, "accuracy")
            if shade_stage:
                span = stage_span(ckcols, task)
                if span:
                    ax.axvspan(span[0], span[1], alpha=0.3, color=STAGE_SHADE, zorder=0)
                    any_shade = True
            # probe 선(깊이 순번 L0~L4 + proj)
            for r in sorted(pidx.get((model, task), []),
                            key=lambda x: DEPTH_ORDER.get(depth_key(x["layer"], slayers), 9)):
                dk = depth_key(r["layer"], slayers); seen_depths.add(dk)
                ax.plot(range(len(ckcols)), disp(r["_s"], ckcols, r["metric"]),
                        color=DEPTH_COLORS.get(dk, "#777777"),
                        marker=DEPTH_MARKERS.get(dk, "o"),
                        markersize=6, lw=2.0, zorder=3)
            # inference 별
            ir = iidx.get((model, task))
            if ir:
                ax.plot(range(len(ckcols)), disp(ir["_s"], ckcols, ir["metric"]),
                        color=INF_COLOR, marker="*", markersize=13, lw=2.2, zorder=5)
            ch = chance.get(task)
            if ch is not None:
                ax.axhline(ch, color=CHANCE_COLOR, ls=":", lw=1.4, zorder=2)
            ax.set_xticks(pos)
            ax.set_xticklabels(ticklab, fontsize=FS_TICK - 4, fontweight="bold")
            ax.get_xticklabels()[0].set_ha("right")
            ax.set_xlim(-0.5, len(ckcols) - 0.5)
            ax.set_ylim(0, 1.05); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.tick_params(axis="y", labelsize=FS_YTICK, labelleft=True)
            plt.setp(ax.get_yticklabels(), fontweight="bold")
            ax.set_ylabel(f"{tlabel} ({metric_abbr(metric)})", fontsize=FS_LABEL,
                          fontweight="bold", labelpad=8)
            ax.grid(axis="y", alpha=0.15, ls=":")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for idx in range(ncol, nbr * wrap):
            axmap[(bi, idx // wrap, idx % wrap)].axis("off")

    # (a)/(b) 모델 태그: 각 블록 좌상단 1회
    for bi, (tag, model, ckcols, pidx, iidx) in enumerate(BLOCKS):
        p = block_top_ax[bi].get_position()
        fig.text(p.x0 + tag_dx, p.y1 + tag_dy, tag, ha="left", va="bottom",
                 fontsize=FS_TAG, fontweight="bold")

    # layer_legend=True 면 L0~L4 대신 실제 번호로 라벨. 색은 모델 공통이라 서로 다른
    # 레이어셋(예: LLaVA 0/8/16/24/32, OpenFlamingo 0/6/12/18/24, Qwen 0/7/14/21/28)을 한 항목에 병기.
    leg_groups, leg_title = layer_legend_groups(models, model_layers)
    def leg_label(d):
        if not layer_legend:
            return d
        if d == "proj":
            return "Projector / Resampler"
        i = int(d[1:])
        return "Layer " + " / ".join(str(ls[i]) for ls in leg_groups)
    if show_legend:
        order = sorted(seen_depths, key=lambda d: DEPTH_ORDER.get(d, 9))
        handles = [mlines.Line2D([0], [0], color=DEPTH_COLORS.get(d, "#777777"),
                                 marker=DEPTH_MARKERS.get(d, "o"), markersize=11, lw=2.5,
                                 label=leg_label(d)) for d in order]
        handles.append(mlines.Line2D([0], [0], color=INF_COLOR, marker="*", markersize=15, lw=2.5, label="Inference"))
        handles.append(mlines.Line2D([0], [0], color=CHANCE_COLOR, ls=":", lw=1.8, label="Chance"))
        if any_shade:
            handles.append(mpatches.Patch(color=STAGE_SHADE, alpha=0.5, label="Training stage"))
        ncol_leg = math.ceil(len(handles) / 2)
        leg = fig.legend(handles=handles, loc="lower center", ncol=ncol_leg,
                         prop={"size": FS_LEGEND, "weight": "bold"},
                         bbox_to_anchor=(0.5, legend_y), frameon=True, edgecolor="#cccccc",
                         title=(leg_title if layer_legend else None))
        if layer_legend and leg.get_title():
            leg.get_title().set_fontsize(FS_LEGEND)
            leg.get_title().set_fontweight("bold")

    save_fig(fig, out_stem, dpi=110)
    plt.close(fig)
    print(f"Saved -> {out_stem}.png / .pdf   ({ncol} tasks, wrap={wrap}, {B} blocks)")
