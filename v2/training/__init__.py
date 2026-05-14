"""
training/__init__ 故意保持空,避免 trl/transformers 的 eager import 在本地测试机
(没装 trl)上 break 整个子模块。需要谁直接从子模块导入:

    from v2.training.pair_builder import sample_pair, PairThresholds
    from v2.training.dpo_dataset import build_dpo_dataset
    from v2.training.sft_dataset import build_sft_dataset
    from v2.training.sft_collator import DynamicSFTCollator   # 需要 trl
    from v2.training.sft_train import main                    # 需要 trl + transformers
"""
