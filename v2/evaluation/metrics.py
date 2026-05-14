"""
评测指标:pass@k(strict)+ mean_pass_ratio + 平均 runtime + 平均 algo_score。

输入:eval_sampling 产出的 exec.jsonl;可选 scores.jsonl(若启用 judge)。
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def estimate_pass_at_k(n_total: np.ndarray, n_correct: np.ndarray, k: int) -> np.ndarray:
    """无偏 pass@k 估计(Codex 论文公式)。"""
    def _est(n: int, c: int, kk: int) -> float:
        if n - c < kk:
            return 1.0
        return 1.0 - float(np.prod(1.0 - kk / np.arange(n - c + 1, n + 1)))
    return np.array([_est(int(n), int(c), k) for n, c in zip(n_total, n_correct)])


def compute_metrics(exec_path: str, scores_path: str | None = None, ks=(1, 10, 100)):
    by_task = defaultdict(list)
    with open(exec_path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            by_task[e["task_id"]].append(e)

    total: list[int] = []
    correct: list[int] = []
    mean_pass_ratios: list[float] = []
    runtimes: list[float] = []

    for tid, es in by_task.items():
        total.append(len(es))
        correct.append(sum(1 for e in es if e["pass_ratio"] == 1.0))
        mean_pass_ratios.append(np.mean([e["pass_ratio"] for e in es]))
        for e in es:
            if e["pass_ratio"] == 1.0 and e.get("runtime_ns_mean") is not None:
                runtimes.append(e["runtime_ns_mean"])

    total_arr = np.array(total)
    correct_arr = np.array(correct)
    out = {}
    for k in ks:
        if (total_arr >= k).all():
            out[f"pass@{k}"] = float(estimate_pass_at_k(total_arr, correct_arr, k).mean())
    out["mean_pass_ratio"] = float(np.mean(mean_pass_ratios))
    if runtimes:
        out["mean_runtime_ns"] = float(np.mean(runtimes))
        out["median_runtime_ns"] = float(np.median(runtimes))
    out["n_problems"] = len(by_task)
    out["n_solutions_per_problem_avg"] = float(np.mean(total))

    if scores_path and os.path.exists(scores_path):
        algos = []
        with open(scores_path, "r", encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                if s.get("scores") and s["scores"].get("parsable"):
                    algos.append(s["scores"]["final"])
        if algos:
            out["mean_algo_final"] = float(np.mean(algos))
            out["n_scored"] = len(algos)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec_path", required=True)
    ap.add_argument("--scores_path", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    m = compute_metrics(args.exec_path, args.scores_path)
    text = json.dumps(m, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
