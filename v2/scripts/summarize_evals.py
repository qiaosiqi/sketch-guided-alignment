"""
汇总所有 eval 目录的 metrics.json,打印完整对比表。

用法:
    python -m v2.scripts.summarize_evals \
        --evals_dir /root/shared-nvme/work/out/evals

输出:
    - 控制台对比表(pass@k / mean_pass_ratio / runtime)
    - --out 指定 JSON 时同时输出机器可读版本
"""
from __future__ import annotations
import argparse
import json
import os


MODELS = ["base", "sft_alg_top25", "dpo_pvf", "dpo_qvs", "dpo_gvb", "dpo_all"]


def load_metrics(evals_dir: str, name: str) -> dict | None:
    p = os.path.join(evals_dir, name, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def fmt(v, pct=False, ms=False, dash="—"):
    if v is None:
        return dash
    if pct:
        return f"{v*100:.2f}%"
    if ms:
        return f"{v/1e6:.3f}ms"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals_dir", default="/root/shared-nvme/work/out/evals")
    ap.add_argument("--out", default=None, help="可选:输出 JSON 路径")
    args = ap.parse_args()

    rows = {}
    for name in MODELS:
        m = load_metrics(args.evals_dir, name)
        rows[name] = m

    # ---- 打印表格 ----
    cols = [
        ("pass@1",           lambda m: fmt(m.get("pass@1"),               pct=True)),
        ("pass@10",          lambda m: fmt(m.get("pass@10"),              pct=True)),
        ("mean_pass_ratio",  lambda m: fmt(m.get("mean_pass_ratio"),      pct=True)),
        ("median_runtime",   lambda m: fmt(m.get("median_runtime_ns"),    ms=True)),
        ("mean_runtime",     lambda m: fmt(m.get("mean_runtime_ns"),      ms=True)),
        ("n_problems",       lambda m: fmt(m.get("n_problems"))),
        ("n_sols_avg",       lambda m: fmt(m.get("n_solutions_per_problem_avg"))),
    ]

    col_w = 16
    name_w = 18

    header = f"{'model':<{name_w}}" + "".join(f"{c[0]:>{col_w}}" for c in cols)
    print(header)
    print("-" * len(header))

    for name in MODELS:
        m = rows[name]
        if m is None:
            row = f"{'[missing]':>{col_w}}" * len(cols)
        else:
            row = "".join(f"{c[1](m):>{col_w}}" for c in cols)
        print(f"{name:<{name_w}}{row}")

    print()
    print("Note: runtime 只统计 pass_ratio==1.0 的解。base/sft pass@1 极低,")
    print("      全通解数量不足,runtime 字段可能缺失或不可靠。")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
