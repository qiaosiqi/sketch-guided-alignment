"""training/dpo_dataset.py 的 smoke tests。

验证动态 transform 能产出 (prompt, chosen, rejected) 三元组,
而且 chosen != rejected。
"""
from v2.training.pair_builder import PairThresholds
from v2.training.dpo_dataset import build_dpo_dataset


def test_dpo_dataset_dynamic_yields_triples(merged_two_problems):
    th = PairThresholds()
    ds = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="pvf",
        thresholds=th, static=False, seed=1,
    )
    assert len(ds) > 0
    row = ds[0]
    assert "prompt" in row
    assert "chosen" in row
    assert "rejected" in row
    assert row["chosen"] != row["rejected"]
    assert "### Sketch" in row["prompt"]
    assert "### Code" in row["chosen"]


def test_dpo_dataset_static(merged_two_problems):
    th = PairThresholds()
    ds = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="pvf",
        thresholds=th, static=True, seed=1,
    )
    # static 模式直接物化 prompt/chosen/rejected
    assert set(ds.column_names) >= {"prompt", "chosen", "rejected"}
    row = ds[0]
    assert row["chosen"] != row["rejected"]


def test_dpo_dataset_gvb_filters_unyieldable(merged_two_problems):
    th = PairThresholds()
    ds = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="gvb",
        thresholds=th, static=False, seed=1,
    )
    # 两题都应该 yieldable (p1 有 score 9/8.5 vs 3, p2 有 score 7.5 vs 2)
    assert len(ds) == 2


def test_dpo_dataset_qvs_requires_full_pass(merged_two_problems):
    th = PairThresholds()
    ds = build_dpo_dataset(
        merged_path=str(merged_two_problems), task="qvs",
        thresholds=th, static=False, seed=1,
    )
    # p1 有 2 个 full-pass 解(1000ns + 2000ns),p2 只有 1 个 → p2 应被过滤
    assert len(ds) == 1
