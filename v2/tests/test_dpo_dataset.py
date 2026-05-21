"""training/dpo_dataset.py 的 smoke tests。

验证 K-pair 静态扩展能产出 (prompt, chosen, rejected) 三元组,且:
- chosen != rejected
- K=1 时每题恰好 1 行,K>1 时 N×K 行
- 同 seed 确定性,不同 seed 产物不同
- task_yieldable 过滤生效
"""
from v2.training.pair_builder import PairThresholds
from v2.training.dpo_dataset import build_dpo_dataset


def test_dpo_dataset_static_default(merged_two_problems):
    """K=1: 每题恰好 1 个 pair。PvF 下 p1 yieldable,p2 不可(无 0.0)。"""
    th = PairThresholds()
    ds = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="pvf",
        thresholds=th, pairs_per_problem=1, seed=1,
    )
    assert len(ds) == 1
    row = ds[0]
    assert "prompt" in row
    assert "chosen" in row
    assert "rejected" in row
    assert row["chosen"] != row["rejected"]
    assert "### Sketch" in row["prompt"]
    assert "### Code" in row["chosen"]


def test_dpo_dataset_k_pair_expansion(merged_two_problems):
    """K=5: 1 题 yieldable × 5 = 5 行。所有行 prompt 相同(同一题),但允许 pair 重复。"""
    th = PairThresholds()
    K = 5
    ds = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="pvf",
        thresholds=th, pairs_per_problem=K, seed=1,
    )
    assert len(ds) == K   # 1 题 × K
    prompts = set(ds["prompt"])
    assert len(prompts) == 1   # 都从同一题来


def test_dpo_dataset_k_pair_multi_problem(merged_two_problems):
    """GvB 下 p1 + p2 均 yieldable,K=3 → 2 × 3 = 6 行。"""
    th = PairThresholds()
    K = 3
    ds = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="gvb",
        thresholds=th, pairs_per_problem=K, seed=1,
    )
    assert len(ds) == 2 * K


def test_dpo_dataset_determinism(merged_two_problems):
    """同 seed 产物完全一致。"""
    th = PairThresholds()
    ds1 = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="gvb",
        thresholds=th, pairs_per_problem=5, seed=42,
    )
    ds2 = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="gvb",
        thresholds=th, pairs_per_problem=5, seed=42,
    )
    assert len(ds1) == len(ds2)
    for r1, r2 in zip(ds1, ds2):
        assert r1["chosen"] == r2["chosen"]
        assert r1["rejected"] == r2["rejected"]


def test_dpo_dataset_different_seeds_differ(merged_two_problems):
    """不同 seed 应产出至少部分不同的扩展(候选池有 4 种 G/B 组合,概率上必不全同)。"""
    th = PairThresholds()
    ds1 = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="gvb",
        thresholds=th, pairs_per_problem=5, seed=1,
    )
    ds2 = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="gvb",
        thresholds=th, pairs_per_problem=5, seed=7,
    )
    differs = any(
        r1["chosen"] != r2["chosen"] or r1["rejected"] != r2["rejected"]
        for r1, r2 in zip(ds1, ds2)
    )
    assert differs


def test_dpo_dataset_gvb_filters_unyieldable(merged_two_problems):
    """p1: score [9, 8.5, 5, 3] 都满足 pass_ratio≥0.5 → G/B 都有;
       p2: pass≥0.5 的有 score [7.5, 2] → G=7.5,B=2,也 yieldable。两题都过。"""
    th = PairThresholds()
    ds = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="gvb",
        thresholds=th, pairs_per_problem=1, seed=1,
    )
    assert len(ds) == 2


def test_dpo_dataset_qvs_requires_full_pass(merged_two_problems):
    """p1 有 2 个 full-pass 解(1000ns + 2000ns) → QvS yieldable;
       p2 只 1 个 full-pass → 不可。"""
    th = PairThresholds()
    ds = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="qvs",
        thresholds=th, pairs_per_problem=1, seed=1,
    )
    assert len(ds) == 1
