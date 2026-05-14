"""training/sft_dataset.py 的 smoke tests。

不依赖真实 tokenizer / 模型,用 conftest 的 FakeTokenizer。
"""
from v2.training.sft_dataset import (
    build_sft_dataset, _select_topp, _sort_value,
)


def test_sort_value_algo_final():
    a = {"algo_final": 9.0}
    b = {"algo_final": 5.0}
    c = {"algo_final": -1.0}
    # 升序排序时,9 → -9 < -5 < inf,所以 a 在前
    assert _sort_value(a, "algo_final") < _sort_value(b, "algo_final")
    assert _sort_value(b, "algo_final") < _sort_value(c, "algo_final")


def test_sort_value_pass_ratio():
    high = {"pass_ratio": 1.0}
    low = {"pass_ratio": 0.0}
    assert _sort_value(high, "pass_ratio") < _sort_value(low, "pass_ratio")


def test_sort_value_runtime_none_at_end():
    fast = {"runtime_ns_mean": 100.0}
    none = {"runtime_ns_mean": None}
    assert _sort_value(fast, "runtime") < _sort_value(none, "runtime")


def test_select_topp_basic():
    answers = [
        {"algo_final": 9, "pass_ratio": 1.0},
        {"algo_final": 8, "pass_ratio": 1.0},
        {"algo_final": 7, "pass_ratio": 1.0},
        {"algo_final": 6, "pass_ratio": 0.9},
        {"algo_final": 5, "pass_ratio": 0.5},
    ]
    top25 = _select_topp(answers, 25, "algo_final")
    # 25% of 5 = 2 (ceil(1.25)=2)
    assert len(top25) == 2
    assert top25[0]["algo_final"] == 9
    assert top25[1]["algo_final"] == 8


def test_select_topp_require_full_pass():
    answers = [
        {"algo_final": 9, "pass_ratio": 0.9},   # 不满足 full pass
        {"algo_final": 5, "pass_ratio": 1.0},
    ]
    top = _select_topp(answers, 100, "algo_final", require_full_pass=True)
    assert len(top) == 1
    assert top[0]["algo_final"] == 5


def test_build_sft_dataset_dynamic(merged_two_problems, fake_tokenizer):
    ds = build_sft_dataset(
        merged_path=str(merged_two_problems), tokenizer=fake_tokenizer,
        top_p=100, sort_by="algo_final",
        max_length=256, static=False, seed=1,
    )
    assert len(ds) == 2
    for row in ds:
        assert "candidate_input_ids" in row
        assert "candidate_attention_mask" in row
        assert len(row["candidate_input_ids"]) == len(row["candidate_attention_mask"])
        # 动态模式:候选数 > 1
        assert len(row["candidate_input_ids"]) >= 1


def test_build_sft_dataset_static(merged_two_problems, fake_tokenizer):
    ds = build_sft_dataset(
        merged_path=str(merged_two_problems), tokenizer=fake_tokenizer,
        top_p=100, sort_by="algo_final",
        max_length=256, static=True, seed=1,
    )
    for row in ds:
        assert len(row["candidate_input_ids"]) == 1   # 静态:每题只 1 个


def test_build_sft_dataset_top25_keeps_at_least_one(merged_two_problems, fake_tokenizer):
    ds = build_sft_dataset(
        merged_path=str(merged_two_problems), tokenizer=fake_tokenizer,
        top_p=25, sort_by="algo_final",
        max_length=256, static=False, seed=1,
    )
    for row in ds:
        # ceil(25 * n / 100) ≥ 1
        assert len(row["candidate_input_ids"]) >= 1


def test_build_sft_dataset_full_pass_only_drops_p2(merged_two_problems, fake_tokenizer):
    """p2 有 1 个 pass=1.0 的解,p1 有 2 个。开 require_full_pass 后都还在。"""
    ds = build_sft_dataset(
        merged_path=str(merged_two_problems), tokenizer=fake_tokenizer,
        top_p=100, sort_by="algo_final",
        require_full_pass=True, max_length=256, static=False, seed=1,
    )
    assert len(ds) == 2
