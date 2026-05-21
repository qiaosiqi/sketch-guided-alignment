"""training/pair_builder.py 的 smoke tests。

测试 PvF / QvS / GvB / all 四种 pair 构造在各种候选分布下的行为。
"""
import random

import pytest

from v2.training.pair_builder import (
    PairThresholds, sample_pair, task_yieldable,
)


def _a(pass_ratio: float, runtime: float | None, algo_final: float) -> dict:
    return {
        "sketch": "s", "code": "c",
        "pass_ratio": pass_ratio, "runtime_ns_mean": runtime, "algo_final": algo_final,
    }


@pytest.fixture
def th():
    return PairThresholds()


@pytest.fixture
def rng():
    return random.Random(0)


# ============================================================
# PvF (二元正确性主信号)
# ============================================================

def test_pvf_basic(th, rng):
    cands = [
        _a(1.0, 100.0, 9),
        _a(0.9, None, 7),     # 部分通过,既不算 pass 也不算 fail
        _a(0.0, None, -1),
    ]
    pair = sample_pair(cands, "pvf", th, rng)
    assert pair is not None
    chosen, rejected = pair
    assert chosen["pass_ratio"] == 1.0
    assert rejected["pass_ratio"] == 0.0


def test_pvf_rejects_partial(th, rng):
    """部分通过的解既不是 pass=1.0 也不是 fail=0.0,PvF 不应采纳。"""
    cands = [_a(0.95, 100.0, 8), _a(0.05, None, -1)]
    assert sample_pair(cands, "pvf", th, rng) is None
    assert not task_yieldable(cands, "pvf", th)


def test_pvf_unyieldable_no_pass(th, rng):
    cands = [_a(0.0, None, -1), _a(0.0, None, -1)]
    assert sample_pair(cands, "pvf", th, rng) is None


def test_pvf_unyieldable_no_fail(th, rng):
    cands = [_a(1.0, 100.0, 9), _a(1.0, 200.0, 8)]
    assert sample_pair(cands, "pvf", th, rng) is None


def test_pvf_thresholds_ignored(th, rng):
    """PvF 不应受 θ_pass_gvb / τ 影响 —— 它的定义就是 1.0 vs 0.0,不参数化。"""
    weird = PairThresholds(theta_pass_gvb=0.0, tau=0.0)
    cands = [_a(1.0, 100.0, 9), _a(0.0, None, -1)]
    pair = sample_pair(cands, "pvf", weird, rng)
    assert pair is not None
    assert pair[0]["pass_ratio"] == 1.0 and pair[1]["pass_ratio"] == 0.0


# ============================================================
# QvS
# ============================================================

def test_qvs_basic(th, rng):
    cands = [
        _a(1.0, 100.0, 9),    # quick
        _a(1.0, 500.0, 9),
        _a(1.0, 900.0, 8),    # slow
    ]
    pair = sample_pair(cands, "qvs", th, rng)
    assert pair is not None
    chosen, rejected = pair
    assert chosen["runtime_ns_mean"] < rejected["runtime_ns_mean"]


def test_qvs_requires_full_pass(th, rng):
    cands = [_a(0.95, 100.0, 9), _a(0.9, 500.0, 8)]
    # 都不是 1.0,QvS 无效
    assert sample_pair(cands, "qvs", th, rng) is None


def test_qvs_runtime_missing_ignored(th, rng):
    cands = [
        _a(1.0, 100.0, 9),
        _a(1.0, None, 8),       # 无 runtime,应被忽略
        _a(1.0, 800.0, 7),
    ]
    pair = sample_pair(cands, "qvs", th, rng)
    assert pair is not None
    for s in pair:
        assert s["runtime_ns_mean"] is not None


# ============================================================
# GvB
# ============================================================

def test_gvb_basic(th, rng):
    cands = [
        _a(1.0, 100.0, 8.5),    # G
        _a(0.9, None, 7.0),     # G (pass_ratio >= 0.5)
        _a(0.85, None, 4.0),    # B
        _a(0.5, None, 3.0),     # B (pass_ratio = 0.5,刚好达标,算法分低)
    ]
    pair = sample_pair(cands, "gvb", th, rng)
    assert pair is not None
    chosen, rejected = pair
    assert chosen["algo_final"] >= th.tau
    assert rejected["algo_final"] < th.tau
    assert chosen["pass_ratio"] >= th.theta_pass_gvb
    assert rejected["pass_ratio"] >= th.theta_pass_gvb


def test_gvb_no_bad(th, rng):
    cands = [_a(1.0, 100.0, 9), _a(0.9, None, 8)]
    assert sample_pair(cands, "gvb", th, rng) is None


def test_gvb_no_good(th, rng):
    cands = [_a(1.0, 100.0, 3), _a(0.85, None, 4)]
    assert sample_pair(cands, "gvb", th, rng) is None


def test_gvb_unscored_excluded(th, rng):
    cands = [
        _a(1.0, 100.0, -1),   # 未评分,不能参与 GvB
        _a(0.85, None, 4),
    ]
    assert sample_pair(cands, "gvb", th, rng) is None


# ============================================================
# all
# ============================================================

def test_all_falls_back_to_pvf(th, rng):
    """没有 QvS(都没 runtime)也没 GvB(都没 score)时,all 应该靠 PvF 找到一对。"""
    cands = [
        _a(1.0, None, -1),    # full pass,未评分,无 runtime → 只 PvF 可用
        _a(0.0, None, -1),
    ]
    pair = sample_pair(cands, "all", th, rng)
    assert pair is not None
    chosen, rejected = pair
    assert chosen["pass_ratio"] == 1.0 and rejected["pass_ratio"] == 0.0


def test_all_unyieldable_all_partial(th, rng):
    """全 partial:PvF 不行(没 1.0/0.0)、QvS 不行(没 1.0)、GvB 不行(没 score)→ all 失败。"""
    cands = [_a(0.5, None, -1), _a(0.55, None, -1)]
    assert sample_pair(cands, "all", th, rng) is None
