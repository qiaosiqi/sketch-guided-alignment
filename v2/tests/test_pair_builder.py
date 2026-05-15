"""training/pair_builder.py 的 smoke tests。

测试 HvL / QvS / GvB / all 四种 pair 构造在各种候选分布下的行为。
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
# HvL
# ============================================================

def test_hvl_basic(th, rng):
    cands = [
        _a(1.0, 100.0, 9),
        _a(0.9, 200.0, 7),
        _a(0.1, None, -1),
    ]
    pair = sample_pair(cands, "hvl", th, rng)
    assert pair is not None
    chosen, rejected = pair
    assert chosen["pass_ratio"] >= th.theta_high
    assert rejected["pass_ratio"] <= th.theta_low


def test_hvl_unyieldable_all_high(th, rng):
    cands = [_a(1.0, 100.0, 9), _a(0.95, 200.0, 8)]
    assert sample_pair(cands, "hvl", th, rng) is None
    assert not task_yieldable(cands, "hvl", th)


def test_hvl_unyieldable_all_mid(th, rng):
    # 全在 [0.3, 0.7) 之间
    cands = [_a(0.5, None, -1), _a(0.6, None, -1)]
    assert sample_pair(cands, "hvl", th, rng) is None


# ============================================================
# PvF (binary baseline)
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
    """PvF 不应受 θ_high / θ_low / τ 影响 —— 这是它作为 ablation 的关键属性。"""
    from v2.training.pair_builder import PairThresholds
    weird = PairThresholds(theta_high=0.01, theta_low=0.99, theta_pass_gvb=0.0, tau=0.0)
    cands = [_a(1.0, 100.0, 9), _a(0.0, None, -1)]
    pair = sample_pair(cands, "pvf", weird, rng)
    assert pair is not None
    assert pair[0]["pass_ratio"] == 1.0 and pair[1]["pass_ratio"] == 0.0


def test_all_excludes_pvf(rng):
    """all 只回退到 hvl/qvs/gvb;只有 PvF 可解时,all 应该失败。"""
    from v2.training.pair_builder import PairThresholds
    # 默认阈值下 PvF 成功 → HvL 必然也成功(1.0≥0.7、0.0≤0.3),无法区分。
    # 故把 HvL 阈值推到 > 1.0 / < 0.0 让 HvL 永远 unyieldable;PvF 仍能配出对。
    weird = PairThresholds(theta_high=1.5, theta_low=-0.5, theta_pass_gvb=0.5, tau=6.0)
    cands = [_a(1.0, None, -1), _a(0.0, None, -1)]
    assert sample_pair(cands, "pvf", weird, rng) is not None
    # HvL 阈值不可达、QvS 无 runtime、GvB 无评分 → all 应失败
    assert sample_pair(cands, "all", weird, rng) is None


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

def test_all_falls_back(th, rng):
    """没有 QvS 也没 GvB 时,all 应该从 HvL 找到一对。"""
    cands = [
        _a(0.95, None, -1),   # 高 pass,未评分,runtime 缺失 → 只 HvL 可用
        _a(0.1, None, -1),
    ]
    pair = sample_pair(cands, "all", th, rng)
    assert pair is not None


def test_all_unyieldable(th, rng):
    cands = [_a(0.5, None, -1), _a(0.55, None, -1)]
    assert sample_pair(cands, "all", th, rng) is None
