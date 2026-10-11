"""curriculum probing figure — 모델별 개별 PDF/PNG.
benchmark_probing_permodel.run() 을 재사용. 각 모델당 (a)Fwd,(b)Rev 를 한 그림에 붙이고
모델마다 자체 legend(모델명 제목 + 실제 레이어 번호, openflamingo=Resampler).
출력: figure/curriculum_probing_<model>.pdf / .png"""
import _csvfig as F
from benchmark_probing_permodel import run, PROBES
from curriculum_probing import TASKS, CHANCE

if __name__ == "__main__":
    for probe, suf in PROBES:
        run("curriculum", TASKS, CHANCE, F.EXP_DIR["exp2"] + "/curriculum_probing" + suf, probe=probe)
