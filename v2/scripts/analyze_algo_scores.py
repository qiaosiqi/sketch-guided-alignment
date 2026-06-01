"""
提炼 test 集 rubric 打分结果:每模型 algo_final / sketch / code / 9 维均值,
+ 按问题配对的 bootstrap 置信区间,判断各模型相对 baseline 的提升是否显著。

前置:对每个模型的子采样目录跑过 04_annotate,产出
    {evals_dir}/{model}/{sub_subdir}/scores.jsonl

统计设计(为什么这么算):
    - 选题在所有模型间是**配对**的(同一批 task_id),所以用**按问题配对**的 bootstrap:
      对每个问题先取该模型已评分解的均值(problem-level),再在问题层面重采样。
      这比"对所有解拉平求均值"更稳:不被某些题解数多寡带偏,且和配对差分一致。
    - 配对差分(model − baseline)只在两者都有可解析评分的问题上算。
    - CI 排除 0 ⇒ 在 bootstrap 意义下显著。注意:judge 与训练信号同源(GLM-4-Air),
      rubric 提升是"信号已学进去"的证据,非独立质量证据——解读见论文 caveat。

用法:
    python -m v2.scripts.analyze_algo_scores \
        --evals_dir /root/shared-nvme/work/out/evals \
        --models base sft_alg_top25 dpo_pvf dpo_qvs dpo_gvb dpo_all \
        --baseline base --out /root/shared-nvme/work/out/evals/algo_summary.json
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

SKETCH_DIMS = ["S1_correctness", "S2_specificity", "S3_complexity_awareness", "S4_edge_coverage"]
CODE_DIMS = ["C1_faithfulness", "C2_time_complexity", "C3_space_complexity",
             "C4_readability", "C5_edge_handling"]


def load_scores(path: str) -> dict[str, list[dict]]:
    """task_id -> [ {final, sketch_mean, code_mean, dims...}, ... ](仅 parsable)。"""
    by_task: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            sc = o.get("scores") or {}
            if not sc.get("parsable"):
                continue
            sk, cd = sc["sketch"], sc["code"]
            rec = {"final": sc["final"]}
            for d in SKETCH_DIMS:
                rec[d] = sk[d]
            for d in CODE_DIMS:
                rec[d] = cd[d]
            rec["sketch_mean"] = float(np.mean([sk[d] for d in SKETCH_DIMS]))
            rec["code_mean"] = float(np.mean([cd[d] for d in CODE_DIMS]))
            by_task.setdefault(o["task_id"], []).append(rec)
    return by_task


def problem_level(by_task: dict[str, list[dict]], field: str) -> dict[str, float]:
    """每问题在该字段上的均值。"""
    return {t: float(np.mean([r[field] for r in recs])) for t, recs in by_task.items()}


def boot_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator, lo=2.5, hi=97.5):
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))


def paired_diff(model_pm: dict[str, float], base_pm: dict[str, float],
                n_boot: int, rng: np.random.Generator):
    common = sorted(set(model_pm) & set(base_pm))
    if not common:
        return None
    d = np.array([model_pm[t] - base_pm[t] for t in common])
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # 双侧 bootstrap p:差分穿过 0 的比例
    frac_le0 = float(np.mean(boot <= 0))
    p = 2.0 * min(frac_le0, 1.0 - frac_le0)
    return {
        "n_common": n,
        "mean_diff": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "p_boot": min(p, 1.0),
        "significant": bool(lo > 0 or hi < 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals_dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--sub_subdir", default="sub")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data: dict[str, dict] = {}
    for m in args.models:
        path = os.path.join(args.evals_dir, m, args.sub_subdir, "scores.jsonl")
        if not os.path.exists(path):
            print(f"[warn] 缺少 {path},跳过 {m}")
            continue
        by_task = load_scores(path)
        n_sols = sum(len(v) for v in by_task.values())
        pm_final = problem_level(by_task, "final")
        pm_sketch = problem_level(by_task, "sketch_mean")
        pm_code = problem_level(by_task, "code_mean")
        data[m] = {
            "by_task": by_task,
            "pm_final": pm_final,
            "n_problems": len(by_task),
            "n_sols": n_sols,
            "final": boot_ci(np.array(list(pm_final.values())), args.n_boot, rng),
            "sketch": boot_ci(np.array(list(pm_sketch.values())), args.n_boot, rng),
            "code": boot_ci(np.array(list(pm_code.values())), args.n_boot, rng),
            "dims": {d: float(np.mean([r[d] for recs in by_task.values() for r in recs]))
                     for d in SKETCH_DIMS + CODE_DIMS},
        }

    # ---- 主表:problem-level 均值 + 95% CI ----
    print("\n=== algo scores (problem-level mean, 95% bootstrap CI) ===")
    hdr = f"{'model':<16}{'n_prob':>7}{'n_sol':>7}   {'algo_final':>22}{'sketch':>16}{'code':>16}"
    print(hdr)
    print("-" * len(hdr))
    for m in args.models:
        if m not in data:
            continue
        d = data[m]
        f_m, f_lo, f_hi = d["final"]
        s_m, s_lo, s_hi = d["sketch"]
        c_m, c_lo, c_hi = d["code"]
        print(f"{m:<16}{d['n_problems']:>7}{d['n_sols']:>7}   "
              f"{f_m:>6.3f} [{f_lo:.3f},{f_hi:.3f}]"
              f"{s_m:>7.3f}[{s_lo:.2f},{s_hi:.2f}]"
              f"{c_m:>7.3f}[{c_lo:.2f},{c_hi:.2f}]")

    # ---- 9 维分解 ----
    print("\n=== 9-dim means ===")
    dim_cols = SKETCH_DIMS + CODE_DIMS
    short = [d.split("_")[0] for d in dim_cols]
    print(f"{'model':<16}" + "".join(f"{s:>7}" for s in short))
    for m in args.models:
        if m not in data:
            continue
        print(f"{m:<16}" + "".join(f"{data[m]['dims'][d]:>7.2f}" for d in dim_cols))

    # ---- 配对差分 vs baseline ----
    paired = {}
    if args.baseline in data:
        base_pm = data[args.baseline]["pm_final"]
        print(f"\n=== paired diff in algo_final vs '{args.baseline}' (95% CI; * = CI excludes 0) ===")
        print(f"{'model':<16}{'n_common':>9}{'mean_diff':>11}   {'ci95':>20}{'p_boot':>9}  sig")
        for m in args.models:
            if m == args.baseline or m not in data:
                continue
            res = paired_diff(data[m]["pm_final"], base_pm, args.n_boot, rng)
            paired[m] = res
            if res is None:
                continue
            star = " *" if res["significant"] else ""
            print(f"{m:<16}{res['n_common']:>9}{res['mean_diff']:>+11.4f}   "
                  f"[{res['ci95'][0]:+.4f},{res['ci95'][1]:+.4f}]{res['p_boot']:>9.4f}{star}")

    if args.out:
        dump = {m: {k: v for k, v in d.items() if k != "by_task"} for m, d in data.items()}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"per_model": dump, "paired_vs_baseline": paired,
                       "baseline": args.baseline, "n_boot": args.n_boot}, f,
                      indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
