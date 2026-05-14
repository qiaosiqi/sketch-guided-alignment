"""
从 APPS raw 目录生成过滤后的 train/val/test jsonl,供后续阶段消费。

用法:
    python -m v2.scripts.01_prepare_apps \
        --apps_root /path/to/APPS/raw \
        --out_dir out/apps
"""
from v2.data.apps_loader import main


if __name__ == "__main__":
    main()
