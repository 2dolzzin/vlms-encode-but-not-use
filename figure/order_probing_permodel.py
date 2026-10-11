"""학습 순서(fwd/rev/랜덤3) probing figure — 모델별 개별 PDF/PNG.
각 모델당 (a)Forward (b)Reverse (c)~(e) Random 다섯 블록을 한 그림에. 선=레이어, 별=inference.
출력: figure/curriculum_probing_orders_<model>.*, figure/benchmark_probing_orders_<model>.*"""
import _csvfig as F
from benchmark_probing_permodel import run, PROBES
from curriculum_probing import TASKS as C_TASKS, CHANCE as C_CHANCE
from benchmark_probing import TASKS as B_TASKS, CHANCE as B_CHANCE

if __name__ == "__main__":
    for probe, suf in PROBES:
        run("curriculum", C_TASKS, C_CHANCE, F.EXP_DIR["exp2"] + "/curriculum_probing_orders" + suf,
            variants=F.VARIANTS, tag_labels=True, probe=probe)
        run("benchmark", B_TASKS, B_CHANCE, F.EXP_DIR["exp4"] + "/benchmark_probing_orders" + suf,
            variants=F.VARIANTS, tag_labels=True, probe=probe)
