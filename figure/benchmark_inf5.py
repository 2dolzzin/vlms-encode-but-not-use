"""benchmark inference figure — 가로 5개 버전(wrap=5). benchmark_inf.py 와 데이터 동일, 레이아웃만 5열."""
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
# mmvp 는 pair_accuracy(쌍 둘 다 정답) → chance 1/4.
CHANCE = {"circled_letter": 1/26, "touching_circles": 1/2, "line_plot": 1/3, "mmvp": 1/4}

if __name__ == "__main__":
    F.plot_inf_wrap("benchmark", TASKS, CHANCE, F.EXP_DIR["exp3"] + "/benchmark_inf5", wrap=5,
                    legend_y=-0.13, tag_dx=-0.1, tag_dy=0.03)
