"""curriculum inference figure — 가로 5개 버전(wrap=5). curriculum_inf.py 와 데이터 동일, 레이아웃만 5열."""
import _csvfig as F

TASKS = [
    ("S0_logamr", "S0 LogMAR"),
    ("S1_similarity", "S1 Color Sim"),   # metric=mae → 1-MAE 축
    ("S2_ishihara", "S2 Ishihara"),
    ("S3_naming", "S3 Color Naming"),
    ("S4_depth", "S4 Depth"),
]
CHANCE = {"S0_logamr": 1/26, "S1_similarity": 1-0.176, "S2_ishihara": 1/10,
          "S3_naming": 1/14, "S4_depth": 1/4}

if __name__ == "__main__":
    F.plot_inf_wrap("curriculum", TASKS, CHANCE, F.EXP_DIR["exp1"] + "/curriculum_inf5", wrap=5, shade_stage=True,
                    legend_y=-0.24, tag_dx=-0.1, tag_dy=0.035)
