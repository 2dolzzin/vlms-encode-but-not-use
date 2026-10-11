"""학습 순서(fwd/rev/랜덤3) 비교 inference figure.
  A) curriculum_inf_orders / benchmark_inf_orders : 블록 = 순서(a~e), 선 = 모델, x = 그 순서의 학습 진행
  B) curriculum_inf_bystage / benchmark_inf_bystage : x = 스테이지 이름 고정, 선 = 순서, 패널 = task × 모델
"""
import _csvfig as F
from curriculum_inf import TASKS as C_TASKS, CHANCE as C_CHANCE
from benchmark_inf import TASKS as B_TASKS, CHANCE as B_CHANCE

if __name__ == "__main__":
    # A) 블록형
    F.plot_inf_wrap("curriculum", C_TASKS, C_CHANCE, F.EXP_DIR["exp1"] + "/curriculum_inf_orders", wrap=5, shade_stage=True,
                    variants=F.VARIANTS, tag_labels=True, legend_y=-0.10, tag_dx=-0.1, tag_dy=0.012, block_gap=0.012)
    F.plot_inf_wrap("benchmark", B_TASKS, B_CHANCE, F.EXP_DIR["exp3"] + "/benchmark_inf_orders", wrap=5,
                    variants=F.VARIANTS, tag_labels=True, legend_y=-0.045, tag_dx=-0.1, tag_dy=0.008, block_gap=0.008)
    # B) 스테이지 고정형
    F.plot_inf_bystage("curriculum", C_TASKS, C_CHANCE, F.EXP_DIR["exp1"] + "/curriculum_inf_bystage", shade_stage=True)
    F.plot_inf_bystage("benchmark", B_TASKS, B_CHANCE, F.EXP_DIR["exp3"] + "/benchmark_inf_bystage")
