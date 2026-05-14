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
        _a(0.9, None, 7.0),     # G (pass_ratio >= 0.8)
        _a(0.85, None, 4.0),    # B
        _a(0.5, None, 3.0),     # 不合格,pass_ratio < 0.8
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
