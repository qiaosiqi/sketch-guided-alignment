"""
DPO 偏好对构造。

Tasks:
    pvf  — Pass vs Fail(二元正确性主信号:pass_ratio == 1.0 vs == 0.0)
    qvs  — Quick vs Slow runtime,前提两者 pass_ratio == 1.0
    gvb  — Good vs Bad algorithm score,前提两者 pass_ratio >= θ_pass_gvb
    all  — 从 pvf/qvs/gvb 三类里能构造出哪种就用哪种(随机回退)

设计选择(2026-05-21):partial-credit 形态的 HvL 信号在 StarCoder2-3B × APPS
interview 上分布近似二元(92% pass_ratio=0,4.6% pass_ratio=1,中间几乎为空),
HvL 与 PvF 的偏好对几乎相同。论文卖点收敛到 sketch-guided 两段式 + GvB,
HvL 整条信号通道下线,PvF 升格为唯一的"正确性"任务。

接口:
    sample_pair(candidates, task, thresholds, rng) -> (chosen, rejected) | None

`candidates` 是 merged.jsonl 中一题的 answers 列表。每个 answer 至少含:
    sketch, code, pass_ratio, runtime_ns_mean (可空), algo_final
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Literal


Task = Literal["pvf", "qvs", "gvb", "all"]


@dataclass
class PairThresholds:
    theta_pass_gvb: float = 0.5     # GvB 双方最低 pass_ratio
    tau: float = 6.0                # GvB 算法分阈值


def _sample_two(rng: random.Random, hi: list, lo: list) -> tuple[dict, dict] | None:
    if not hi or not lo:
        return None
    return rng.choice(hi), rng.choice(lo)


def sample_pvf(candidates: list[dict], th: PairThresholds, rng: random.Random):
    """二元正确性信号:chosen.pass_ratio == 1.0 vs rejected.pass_ratio == 0.0.

    `th` 在 PvF 下不参与;保留参数签名只为 sample_pair 统一调度。
    """
    del th
    pass_full = [c for c in candidates if c["pass_ratio"] == 1.0]
    fail_total = [c for c in candidates if c["pass_ratio"] == 0.0]
    return _sample_two(rng, pass_full, fail_total)


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
    if task == "pvf":
        return sample_pvf(candidates, thresholds, rng)
    if task == "qvs":
        return sample_qvs(candidates, thresholds, rng)
    if task == "gvb":
        return sample_gvb(candidates, thresholds, rng)
    if task == "all":
        # 随机次序尝试三种,第一个成功就返回。
        order = ["pvf", "qvs", "gvb"]
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
