"""benchmark(blind) inference figure — csv_out/inf_benchmark_*.csv 기반.
9 task = blind 8개 + line_plot(구 result_origin 소스). 전부 accuracy."""
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
# 명확히 아는 것만. 나머지 blind task 는 클래스 수 가변이라 생략.
# mmvp 는 pair_accuracy(쌍 둘 다 정답) → chance 1/4.
CHANCE = {"circled_letter": 1/26, "touching_circles": 1/2, "line_plot": 1/3, "mmvp": 1/4}

if __name__ == "__main__":
    F.plot_inf_wrap("benchmark", TASKS, CHANCE, F.EXP_DIR["exp3"] + "/benchmark_inf", wrap=3,
                    legend_y=-0.035, tag_dx=-0.1, tag_dy=0.02)
