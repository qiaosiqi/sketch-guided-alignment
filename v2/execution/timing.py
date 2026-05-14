"""
稳定测时:沿用原论文 CoV ≤ 0.1 的循环。
仅在 pass_ratio == 1.0 时调用(失败的解没必要测时)。

测时方式与旧版 utils/execution.py 一致:
- 每轮跑 50 次 exec,记录 process_time_ns
- 计算 mean / std,如果 std/mean ≤ 0.1 就停
- 最多 1000 轮(50000 次执行),超出则视为测时失败
"""
from __future__ import annotations
import numpy as np
from time import process_time_ns
from typing import Callable


def stable_runtime(
    run_once: Callable[[], None],
    inner_repeats: int = 50,
    max_outer: int = 1000,
    cov_threshold: float = 0.1,
) -> tuple[float | None, float | None]:
    """
    重复执行 run_once 直到 std/mean ≤ cov_threshold 或达到上限。

    返回 (mean_ns, std_ns)。若一直没收敛,返回 (None, None)。

    注意:调用者负责给 run_once 加 swallow_io / time_limit 等保护。
    """
    times: list[int] = []
    for outer in range(max_outer):
        times = []
        for _ in range(inner_repeats):
            t0 = process_time_ns()
            run_once()
            times.append(process_time_ns() - t0)
        arr = np.asarray(times, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        if mean > 0 and std / mean <= cov_threshold:
            return mean, std
    return None, None
