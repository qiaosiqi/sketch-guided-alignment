"""
DPO 训练入口。本脚本是 v2.training.dpo_train 的 thin wrapper。

典型用法(5090 ×2 + DeepSpeed ZeRO-3 无 offload):
    deepspeed --num_gpus 2 --module v2.scripts.07_train_dpo \
        --train_merged /data/work/out/datasets/train/merged.jsonl \
        --val_merged /data/work/out/datasets/val/merged.jsonl \
        --model_path /data/work/out/runs/sft_alg_top25/best \
        --output_dir /data/work/out/runs/dpo_gvb_from_sft \
        --task gvb \
        --augment True \
        --ds_config v2/configs/ds_zero3_2gpu.json

如全参数 OOM,加 --use_lora(切回 ZeRO-2 也行,LoRA 显存最省)。
"""
from v2.training.dpo_train import main


if __name__ == "__main__":
    main()
