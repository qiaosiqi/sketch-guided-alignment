"""evaluation/metrics.py 的 smoke tests。"""
import json
from pathlib import Path

import numpy as np

from v2.evaluation.metrics import estimate_pass_at_k, compute_metrics


def test_pass_at_k_all_correct():
    # 10 个全对 → pass@1 = 1.0
    p = estimate_pass_at_k(np.array([10]), np.array([10]), 1)
    assert p[0] == 1.0


def test_pass_at_k_none_correct():
    p = estimate_pass_at_k(np.array([10]), np.array([0]), 1)
    assert p[0] == 0.0


def test_pass_at_k_half():
    # 10 个,5 对:pass@1 = 5/10 = 0.5
    p = estimate_pass_at_k(np.array([10]), np.array([5]), 1)
    assert abs(p[0] - 0.5) < 1e-9


def test_pass_at_k_when_n_minus_c_less_than_k():
    # n=5, c=4, k=3:n-c=1 < k=3,直接返回 1.0
    p = estimate_pass_at_k(np.array([5]), np.array([4]), 3)
    assert p[0] == 1.0


def test_compute_metrics_basic(tmp_path):
    # 2 题,每题 5 个解,task1 全过,task2 全失败
    rows = []
    for i in range(5):
        rows.append({
            "task_id": "t1", "sample_id": i, "code_id": 0,
            "n_tests": 3, "n_passed": 3, "pass_ratio": 1.0,
            "per_test_pass": [True]*3,
            "runtime_ns_mean": 100.0 + i, "runtime_ns_std": 5.0,
        })
    for i in range(5):
        rows.append({
            "task_id": "t2", "sample_id": i, "code_id": 0,
            "n_tests": 3, "n_passed": 0, "pass_ratio": 0.0,
            "per_test_pass": [False]*3,
            "runtime_ns_mean": None, "runtime_ns_std": None,
        })
    exec_path = tmp_path / "exec.jsonl"
    with open(exec_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    m = compute_metrics(str(exec_path), ks=(1, 5))
    # 2 题,平均 pass@1 = (1.0 + 0.0) / 2 = 0.5
    assert abs(m["pass@1"] - 0.5) < 1e-9
    # pass@5: task1 = 1.0, task2 = 0.0 → 平均 0.5
    assert abs(m["pass@5"] - 0.5) < 1e-9
    # 平均 pass_ratio = (1.0 + 0.0) / 2 = 0.5
    assert abs(m["mean_pass_ratio"] - 0.5) < 1e-9
    # runtime 只统计 task1 的 5 个 runtime,平均 102.0
    assert abs(m["mean_runtime_ns"] - 102.0) < 1e-9
    assert m["n_problems"] == 2
