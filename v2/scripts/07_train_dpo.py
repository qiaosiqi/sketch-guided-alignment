"""
DPO 训练入口。本脚本是 v2.training.dpo_train 的 thin wrapper。

典型用法(单卡 V100-32G + DeepSpeed ZeRO-3 全 offload):
    deepspeed --num_gpus 1 -m v2.scripts.07_train_dpo \
        --train_merged out/datasets/train/merged.jsonl \
        --val_merged out/datasets/val/merged.jsonl \
        --model_path out/runs/sft_alg_top25 \
        --output_dir out/runs/dpo_gvb_from_sft \
        --task gvb \
        --augment True \
        --ds_config v2/configs/ds_zero3_offload.json

如全参数 OOM,加 --use_lora。
"""
from v2.training.dpo_train import main


if __name__ == "__main__":
    main()
