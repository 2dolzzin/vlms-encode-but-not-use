"""curriculum probing figure — csv_out/probing_curriculum_*.csv (+inf overlay) 기반."""
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
    for probe, suf in (("mlp", ""), ("linear", "_linear")):
        F.plot_probing_wrap("curriculum", TASKS, CHANCE, F.EXP_DIR["exp2"] + "/curriculum_probing" + suf, wrap=5,
                            shade_stage=True, legend_y=-0.03, tag_dx=-0.05, tag_dy=0.008,
                            layer_legend=True, probe=probe)
