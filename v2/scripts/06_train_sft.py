"""
SFT 训练入口。本脚本只是 v2.training.sft_train 的 thin wrapper。

典型用法(5090 ×2 + DeepSpeed ZeRO-2 无 offload):
    deepspeed --num_gpus 2 -m v2.scripts.06_train_sft \
        --train_merged /data/work/out/datasets/train/merged.jsonl \
        --val_merged /data/work/out/datasets/val/merged.jsonl \
        --model_path /data/models/StarCoder2-3B \
        --output_dir /data/work/out/runs/sft_alg_top25 \
        --sort_by algo_final \
        --top_p 25 \
        --augment True \
        --ds_config v2/configs/ds_zero2_2gpu.json
"""
from v2.training.sft_train import main


if __name__ == "__main__":
    main()
