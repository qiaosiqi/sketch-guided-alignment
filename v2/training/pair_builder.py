"""
DPO 偏好对构造(partial-credit 版)。

Tasks:
    hvl  — High vs Low pass_ratio
    qvs  — Quick vs Slow runtime,前提两者 pass_ratio == 1.0
    gvb  — Good vs Bad algorithm score,前提两者 pass_ratio >= θ_pass_gvb
    all  — 从 hvl/qvs/gvb 三类里能构造出哪种就用哪种(随机回退)

接口:
    sample_pair(candidates, task, thresholds, rng) -> (chosen, rejected) | None

`candidates` 是 merged.jsonl 中一题的 answers 列表。每个 answer 至少含:
    sketch, code, pass_ratio, runtime_ns_mean (可空), algo_final
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Literal


Task = Literal["hvl", "qvs", "gvb", "all"]


@dataclass
class PairThresholds:
    theta_high: float = 0.7         # HvL 高分阈值
    theta_low: float = 0.3          # HvL 低分阈值
    theta_pass_gvb: float = 0.8     # GvB 双方最低 pass_ratio
    tau: float = 6.0                # GvB 算法分阈值


def _sample_two(rng: random.Random, hi: list, lo: list) -> tuple[dict, dict] | None:
    if not hi or not lo:
        return None
    return rng.choice(hi), rng.choice(lo)


def sample_hvl(candidates: list[dict], th: PairThresholds, rng: random.Random):
    hi = [c for c in candidates if c["pass_ratio"] >= th.theta_high]
    lo = [c for c in candidates if c["pass_ratio"] <= th.theta_low]
    return _sample_two(rng, hi, lo)


def sample_qvs(candidates: list[dict], th: PairThresholds, rng: random.Random):
    full = [c for c in candidates if c["pass_ratio"] == 1.0
            and c.get("runtime_ns_mean") is not None]
    if len(full) < 2:
        return None
    full = sorted(full, key=lambda c: c["runtime_ns_mean"])
    # 上半段为 quick,下半段为 slow
    mid = len(full) // 2
    quick = full[:mid] if mid else full[:1]
    slow = full[mid:] if mid else full[1:]
    return _sample_two(rng, quick, slow)


def sample_gvb(candidates: list[dict], th: PairThresholds, rng: random.Random):
    eligible = [c for c in candidates
                if c["pass_ratio"] >= th.theta_pass_gvb
                and c.get("algo_final", -1) >= 0]
    good = [c for c in eligible if c["algo_final"] >= th.tau]
    bad = [c for c in eligible if c["algo_final"] < th.tau]
    return _sample_two(rng, good, bad)


def sample_pair(
    candidates: list[dict],
    task: Task,
    thresholds: PairThresholds,
    rng: Optional[random.Random] = None,
) -> Optional[tuple[dict, dict]]:
    rng = rng or random
    if task == "hvl":
        return sample_hvl(candidates, thresholds, rng)
    if task == "qvs":
        return sample_qvs(candidates, thresholds, rng)
    if task == "gvb":
        return sample_gvb(candidates, thresholds, rng)
    if task == "all":
        # 随机次序尝试三种,第一个成功就返回
        order = ["hvl", "qvs", "gvb"]
        rng.shuffle(order)
        for t in order:
            res = sample_pair(candidates, t, thresholds, rng)  # type: ignore[arg-type]
            if res is not None:
                return res
        return None
    raise ValueError(f"unknown task: {task}")


def task_yieldable(candidates: list[dict], task: Task, thresholds: PairThresholds) -> bool:
    """题级判定:这题对该 task 能不能构造出至少一对。用于过滤数据。"""
    # 用 deterministic rng 检查存在性
    return sample_pair(candidates, task, thresholds, random.Random(0)) is not None
