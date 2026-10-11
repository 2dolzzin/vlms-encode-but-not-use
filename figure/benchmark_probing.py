"""benchmark(blind) probing figure — csv_out/probing_benchmark_*.csv (+inf overlay) 기반.
9 task = blind 8개 + line_plot. 전부 clf(accuracy). OpenFlamingo 포함(blind probing 있음)."""
import _csvfig as F

TASKS = [
    ("circled_letter", "Circled Letter"),
    ("touching_circles", "Touching Circles"),
    ("nested_squares", "Nested Squares"),
    ("olympic_circles", "Olympic Circles"),
    ("olympic_pentagons", "Olympic Pentagons"),
    ("subway_connections", "Subway Conn."),
    ("counting_grid_blank", "Counting (blank)"),
    ("counting_grid_word", "Counting (word)"),
    ("line_plot", "Line Plot"),
    ("mmvp", "MMVP"),
]
# mmvp probing 은 정답문장 146-class 분류 → chance 1/146.
CHANCE = {"circled_letter": 1/26, "touching_circles": 1/2, "line_plot": 1/3, "mmvp": 1/146}

if __name__ == "__main__":
    # 두 장으로 분할: benchmark_probing_1 = LLaVA/Mipha/VILA-3B (legend 없음),
    #               benchmark_probing_2 = OpenFlamingo/Qwen2.5-VL (legend 붙임)
    common = dict(wrap=5, legend_y=-0.005, tag_dx=-0.09, tag_dy=0.0010,
                  gap_ratio=0.08, layer_legend=True)
    for probe, suf in (("mlp", ""), ("linear", "_linear")):
        F.plot_probing_wrap("benchmark", TASKS, CHANCE, F.EXP_DIR["exp4"] + f"/benchmark_probing_1{suf}",
                            models=["llava", "mipha", "vila_3b"], show_legend=False, probe=probe, **common)
        F.plot_probing_wrap("benchmark", TASKS, CHANCE, F.EXP_DIR["exp4"] + f"/benchmark_probing_2{suf}",
                            models=["openflamingo", "qwen25vl_7b"], show_legend=True, probe=probe, **common)
