"""
SFT 训练入口。本脚本只是 v2.training.sft_train 的 thin wrapper。

典型用法(单卡 V100-32G + DeepSpeed ZeRO-3):
    deepspeed --num_gpus 1 -m v2.scripts.06_train_sft \
        --train_merged out/datasets/train/merged.jsonl \
        --val_merged out/datasets/val/merged.jsonl \
        --model_path /path/to/StarCoder2-3B \
        --output_dir out/runs/sft_alg_top25 \
        --sort_by algo_final \
        --top_p 25 \
        --augment True \
        --ds_config v2/configs/ds_zero3_offload.json
"""
from v2.training.sft_train import main


if __name__ == "__main__":
    main()
