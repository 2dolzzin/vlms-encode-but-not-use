# order_runs.py
# 무작위 순서 커리큘럼 런(checkpoint/<Model>_order_<tag>_nohex/) 을 평가 스크립트들이
# 공통으로 다루기 위한 유틸. extract_5_tasks/inf_*.py, extract_benchmark/ben_*.py, export_csv.py 가 쓴다.
#
# 디스크 레이아웃
#   forward/reverse 런 : checkpoint/llava/S2_Ishihara/{checkpoint-1563, end}
#   무작위 순서 런      : checkpoint/llava_order_20431_nohex/1_S2_Ishihara/{checkpoint-1563, end}
#                        (접두사 = 학습 순서, order.txt 에 순서 기록)
#
# 체크포인트 키는 두 경우 모두 "S2_mid" / "S2_end" 로 같다 — 키는 스테이지 정체성만 담고,
# 시점 순서(열 순서)는 ckpt_order() 로 따로 얻는다. 그래서 채점/프로빙 코드는 그대로 호환된다.
import os
import re
import glob

ORDER_DIR_RE = re.compile(r"^(\d+)_(S[0-4])_")      # "1_S2_Ishihara", "3_S0_LogMAR-phi2" ...
STAGE_IDS = ["S0", "S1", "S2", "S3", "S4"]


def is_order_run(root):
    """root 바로 아래에 "<n>_S<k>_..." 폴더가 있으면 무작위 순서 런."""
    return bool(glob.glob(os.path.join(root, "[0-9]_S[0-4]_*")))


def stage_dirs(root, stages, verbose=True):
    """{sid: stage_dir}.
    - 무작위 순서 런: "<n>_S<k>_<name>" 폴더를 학습 순서(n)대로.
    - 그 외: stages 매핑({sid: 폴더명}) 그대로 (forward/reverse 와 동일 동작)."""
    ordered = sorted(glob.glob(os.path.join(root, "[0-9]_S[0-4]_*")),
                     key=lambda d: int(os.path.basename(d).split("_", 1)[0]))
    if not ordered:
        return {sid: os.path.join(root, dname) for sid, dname in stages.items()}
    out = {}
    for d in ordered:
        sid = ORDER_DIR_RE.match(os.path.basename(d)).group(2)
        if sid in out:
            raise RuntimeError(f"{root}: 스테이지 {sid} 폴더가 둘 이상 ({out[sid]}, {d})")
        out[sid] = d
    if verbose:
        print(f"[order] {os.path.basename(root)} 학습순서 = "
              + " → ".join(os.path.basename(d) for d in ordered))
    return out


def read_order(root):
    """학습 순서 [sid, ...]. order.txt 가 있으면 그것을, 없으면 폴더 접두사에서 유도.
    forward/reverse 런(접두사 폴더 없음)은 None."""
    f = os.path.join(root, "order.txt")
    if os.path.exists(f):
        sids = [ln.strip() for ln in open(f) if re.fullmatch(r"S[0-4]", ln.strip())]
        if len(sids) == 5:
            return sids
    if not is_order_run(root):
        return None
    return list(stage_dirs(root, {}, verbose=False).keys())


def ckpt_order(order):
    """학습 순서 [sid,...] → 시점 키 11개 (열 순서). 예) ["S2","S0",..] →
    ["S0_start","S2_mid","S2_end","S0_mid","S0_end", ...]  (S0_start 는 항상 base 모델)"""
    keys = ["S0_start"]
    for sid in order:
        keys += [f"{sid}_mid", f"{sid}_end"]
    return keys


def order_tag(root):
    """checkpoint/llava_order_20431_nohex → "20431". order 런이 아니면 None."""
    m = re.search(r"_order_(\d{5})", os.path.basename(os.path.normpath(root)))
    return m.group(1) if m else None


def discover_order_runs(ckpt_root, model_prefix, key_prefix, out_suffix="", **extra):
    """checkpoint/<model_prefix>_order_*  →  {run_key: dict(root=..., out=..., **extra)}
    예) discover_order_runs(".../checkpoint", "VILA_3B", "vila_3b")
        → {"vila_3b_order_14203": dict(root=".../VILA_3B_order_14203_nohex", out="vila_3b_order_14203")}
    run_key 는 <key_prefix>_order_<tag> 로 통일한다(폴더의 _nohex 접미사는 떼어냄; 무작위 순서 런은 전부 nohex).
    아직 스테이지 폴더가 하나도 없는(막 시작한) 런은 건너뛴다."""
    runs = {}
    for root in sorted(glob.glob(os.path.join(ckpt_root, f"{model_prefix}_order_*"))):
        if not os.path.isdir(root) or not is_order_run(root):
            continue
        tag = order_tag(root)
        if tag is None:
            continue
        key = f"{key_prefix}_order_{tag}"
        if key in runs:                           # 같은 tag 가 둘(예: hex/nohex)이면 폴더 접미사를 살려 구분
            key = f"{key_prefix}_order_" + os.path.basename(root).split("_order_", 1)[1]
        runs[key] = dict(root=root, out=key + out_suffix, **extra)
    return runs


def describe_ckpts(ckpts, order=None):
    """--list 출력용: 시점 키 → 경로 를 학습 순서대로 한 줄씩."""
    keys = ckpt_order(order) if order else list(ckpts)
    lines = [f"  {k:10s} {ckpts[k]}" for k in keys if k in ckpts]
    missing = [k for k in keys if k not in ckpts]
    if missing:
        lines.append(f"  (없음: {', '.join(missing)})")
    return "\n".join(lines)


def order_states(root, base, mid="checkpoint-1563", end="end"):
    """망각 행렬(eval loss) 런처용 상태 목록: [("start", base), ("S2_mid", <dir>/mid), ("S2_end", <dir>/end), ...] 학습 순서대로.
    end="." 이면 스테이지 폴더 자체(OpenFlamingo: flamingo_model.pt 가 폴더에 있음)."""
    out = [("start", base)]
    for sid, d in stage_dirs(root, {}, verbose=False).items():
        out.append((f"{sid}_mid", os.path.join(d, mid)))
        out.append((f"{sid}_end", d if end == "." else os.path.join(d, end)))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="무작위 순서 런 유틸 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("states", help="런처용 'tag|path' 목록 출력 (학습 순서)")
    a.add_argument("root"); a.add_argument("--base", required=True); a.add_argument("--mid", default="checkpoint-1563"); a.add_argument("--end", default="end")
    b = sub.add_parser("root", help="run 키 → 체크포인트 루트 (예: vila_3b_order_14203)")
    b.add_argument("run_key"); b.add_argument("--ckpt_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint"))
    args = ap.parse_args()
    if args.cmd == "states":
        for tag, path in order_states(args.root, args.base, args.mid, args.end):
            print(f"{tag}|{path}")
    else:
        PREFIX = {"llava": "llava", "vila_3b": "VILA_3B", "vila_8b": "VILA_8B", "mipha": "Mipha", "openflamingo": "OpenFlamingo", "qwen25vl_7b": "Qwen25VL_7B"}
        m = re.match(r"^(.*)_order_(\d{5})$", args.run_key)
        if not m or m.group(1) not in PREFIX:
            raise SystemExit(f"order run 키가 아님: {args.run_key}")
        cands = sorted(glob.glob(os.path.join(args.ckpt_root, f"{PREFIX[m.group(1)]}_order_{m.group(2)}*")))
        if not cands:
            raise SystemExit(f"체크포인트 없음: {args.run_key}")
        print(cands[0])
