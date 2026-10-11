"""benchmark(blind) probing figure — 모델별 개별 PDF/PNG.
각 모델당 (a)Fwd, (b)Rev 두 블록을 한 그림에 붙여서 저장.
출력: figure/benchmark_probing_<model>.pdf / .png
기존 benchmark_probing.py(합본)는 그대로 두고, 이 스크립트는 분할본만 생성."""
import os, math
import matplotlib.pyplot as plt
from _csvfig import save_fig, EXP_DIR
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import _csvfig as F
from _csvfig import (MODELS, MODEL_LABELS,
                     DEPTH_COLORS, DEPTH_MARKERS, DEPTH_ORDER,
                     INF_COLOR, CHANCE_COLOR, STAGE_SHADE,
                     FS_LABEL, FS_TICK, FS_TAG, FS_LEGEND, FS_YTICK,
                     tick_info, disp, depth_key, stage_span, metric_abbr, load_probe_dirs)
from benchmark_probing import TASKS, CHANCE

NAME = "benchmark"
OUT_STEM = EXP_DIR["exp4"] + "/benchmark_probing"
WRAP = 5
PANEL_W, PANEL_H = 7.0, 7.6
# 합본과 동일한 태그/범례 미세조정
LEGEND_Y, TAG_DX, TAG_DY = -0.005, -0.09, 0.0010
HSPACE = 0.30           # (a) 2행 및 (a)↔(b) 세로 간격 (붙이려면 낮춰)
LEGEND_TITLE_SEP = 24   # 범례 제목(모델명) ↔ layer 항목 세로 간격(pt). 키우면 더 벌어짐


def run(name, tasks, chance, out_stem, variants=("forward", "reverse"), tag_labels=False, models=None, probe="mlp"):
    """variants: 모델마다 이 순서로 블록((a),(b),…). tag_labels=True 면 태그에 순서 라벨 병기. probe='mlp'|'linear'."""
    DIRS, task_metric, model_layers = load_probe_dirs(name, variants, probe)

    ncol = len(tasks)
    nbr = math.ceil(ncol / WRAP)     # 블록당 행 수

    for model in (models if models is not None else MODELS):
        # 이 모델의 variant 블록들 — spacer 없이 붙임
        BLOCKS = [(f"({letter}) {MODEL_LABELS[model]}" + (f" — {vlabel}" if tag_labels else ""),
                   model, ckcols, pidx, iidx)
                  for letter, vlabel, ckcols, pidx, iidx in DIRS]
        B = len(BLOCKS)
        total_rows = B * nbr
        fig_h = PANEL_H * total_rows
        fig = plt.figure(figsize=(PANEL_W * WRAP, fig_h))
        gs = fig.add_gridspec(total_rows, WRAP, hspace=HSPACE, wspace=0.5,
                              left=0.06, right=0.99, top=0.985, bottom=0.02)
        axmap = {}
        block_top_ax = {}
        grow = 0
        for bi in range(B):
            for r in range(nbr):
                for c in range(WRAP):
                    ax = fig.add_subplot(gs[grow, c])
                    axmap[(bi, r, c)] = ax
                    if r == 0 and c == 0:
                        block_top_ax[bi] = ax
                grow += 1

        seen_depths = set()
        any_shade = False
        for bi, (tag, m, ckcols, pidx, iidx) in enumerate(BLOCKS):
            slayers = model_layers.get(m, [0, 8, 16, 24, 32])
            pos, ticklab = tick_info(ckcols)
            for ti, (task, tlabel) in enumerate(tasks):
                ax = axmap[(bi, ti // WRAP, ti % WRAP)]
                metric = task_metric.get(task, "accuracy")
                span = stage_span(ckcols, task)
                if span:
                    ax.axvspan(span[0], span[1], alpha=0.3, color=STAGE_SHADE, zorder=0)
                    any_shade = True
                for r in sorted(pidx.get((m, task), []),
                                key=lambda x: DEPTH_ORDER.get(depth_key(x["layer"], slayers), 9)):
                    dk = depth_key(r["layer"], slayers); seen_depths.add(dk)
                    ax.plot(range(len(ckcols)), disp(r["_s"], ckcols, r["metric"]),
                            color=DEPTH_COLORS.get(dk, "#777777"),
                            marker=DEPTH_MARKERS.get(dk, "o"),
                            markersize=6, lw=2.0, zorder=3)
                ir = iidx.get((m, task))
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
            for idx in range(ncol, nbr * WRAP):
                axmap[(bi, idx // WRAP, idx % WRAP)].axis("off")

        # (a)/(b) 태그: 각 블록 좌상단 1회
        for bi, (tag, m, ckcols, pidx, iidx) in enumerate(BLOCKS):
            p = block_top_ax[bi].get_position()
            fig.text(p.x0 + TAG_DX, p.y1 + TAG_DY, tag, ha="left", va="bottom",
                     fontsize=FS_TAG, fontweight="bold")

        # 범례: 모델별 각자. title=모델명, layer=해당 모델 실제 번호,
        # proj = openflamingo 는 Resampler / 그 외는 Projector.
        slayers_m = model_layers.get(model, [0, 8, 16, 24, 32])
        def dlabel(d):
            if d == "proj":
                return "Resampler" if model == "openflamingo" else "Projector"
            return f"Layer {slayers_m[int(d[1:])]}"
        order = sorted(seen_depths, key=lambda d: DEPTH_ORDER.get(d, 9))
        handles = [mlines.Line2D([0], [0], color=DEPTH_COLORS.get(d, "#777777"),
                                 marker=DEPTH_MARKERS.get(d, "o"), markersize=11, lw=2.5,
                                 label=dlabel(d)) for d in order]
        handles.append(mlines.Line2D([0], [0], color=INF_COLOR, marker="*", markersize=15, lw=2.5, label="Inference"))
        handles.append(mlines.Line2D([0], [0], color=CHANCE_COLOR, ls=":", lw=1.8, label="Chance"))
        if any_shade:
            handles.append(mpatches.Patch(color=STAGE_SHADE, alpha=0.5, label="Training stage"))
        ncol_leg = math.ceil(len(handles) / 2)
        # loc="upper center" + 축 하단보다 아래 anchor → 그래프와 안 겹치게 완전히 아래에
        leg = fig.legend(handles=handles, loc="upper center", ncol=ncol_leg,
                         prop={"size": FS_LEGEND, "weight": "bold"},
                         bbox_to_anchor=(0.5, -0.05), frameon=True, edgecolor="#cccccc",
                         title=MODEL_LABELS[model])
        if leg.get_title():
            leg.get_title().set_fontsize(FS_LEGEND)
            leg.get_title().set_fontweight("bold")
        # 제목(모델명) ↔ 항목 간격 넓히기
        leg._legend_box.sep = LEGEND_TITLE_SEP
        leg._legend_box.stale = True

        stem = f"{out_stem}_{model}"
        save_fig(fig, stem, dpi=110)
        plt.close(fig)
        print(f"Saved -> {stem}.png / .pdf   ({ncol} tasks, wrap={WRAP}, {B} blocks)")


PROBES = [("mlp", ""), ("linear", "_linear")]   # (probe, 파일 접미사) — MLP / 선형 두 벌 생성

if __name__ == "__main__":
    for probe, suf in PROBES:
        run(NAME, TASKS, CHANCE, OUT_STEM + suf, probe=probe)
