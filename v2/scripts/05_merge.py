"""
入口脚本:把 sample_dir 下所有 jsonl 合并成 merged.jsonl。

用法:
    python -m v2.scripts.05_merge \
        --problems_jsonl out/apps/train.jsonl \
        --sample_dir out/main \
        --out out/datasets/train/merged.jsonl
"""
from v2.merge.build_dataset import main


if __name__ == "__main__":
    main()
