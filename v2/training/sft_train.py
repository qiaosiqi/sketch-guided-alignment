"""
SFT 训练入口。基于 trl 0.11+ 的 SFTTrainer / SFTConfig。

关键设计:
- A800-80G bf16(StarCoder2-3B 原生 bf16;DPO lr=5e-7 在 fp16 下易溢出)
- DeepSpeed ZeRO-2 无 offload (ds_zero2.json);3B 全参 + Adam ≈ 48GB,80G 单卡放得下,
  绝不能 offload 到 18 核 CPU
- gradient_checkpointing=True(可关:80G 单卡显存够;留 True 保守省显存)
- 自定义 DynamicSFTCollator,从每题 top-p% 候选池里每步采一个
- 训练验证集来自 merged.jsonl 的 val 切片(由 prepare_apps 9:1 拆出来)

用法:
    python -m v2.training.sft_train \
        --train_merged out/datasets/train/merged.jsonl \
        --val_merged out/datasets/val/merged.jsonl \
        --model_path /path/to/StarCoder2-3B \
        --output_dir out/runs/sft_alg_top25 \
        --sort_by algo_final \
        --top_p 25 \
        --augment True
"""
from __future__ import annotations
import argparse
import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTTrainer, SFTConfig

from .sft_dataset import build_sft_dataset, response_template_ids
from .sft_collator import DynamicSFTCollator


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_merged", required=True)
    ap.add_argument("--val_merged", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)

    # 数据选择
    ap.add_argument("--sort_by", default="algo_final",
                    choices=["algo_final", "pass_ratio", "runtime"])
    ap.add_argument("--top_p", type=int, default=25, help="保留 top-p% 候选")
    ap.add_argument("--require_full_pass", action="store_true",
                    help="只保留 pass_ratio=1.0 的候选(QvS 风格)")
    ap.add_argument("--augment", type=str, default="True", choices=["True", "False"],
                    help="True = 动态选择(每步随机挑一);False = 静态(每题固定一份)")

    # 训练超参
    ap.add_argument("--learning_rate", type=float, default=5e-7)
    ap.add_argument("--num_train_epochs", type=int, default=10)
    # A800-80G: micro-batch 调大、grad-accum 缩小,有效 batch 仍 = 32(8×4),仅提速
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--ds_config", default=None, help="DeepSpeed config 路径")

    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ---- model + tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=args.max_length,
        use_fast=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
    )
    model.config.use_cache = False  # 与 gradient_checkpointing 冲突,必须关

    # ---- datasets ----
    train_ds = build_sft_dataset(
        merged_path=args.train_merged, tokenizer=tokenizer,
        top_p=args.top_p, sort_by=args.sort_by,
        require_full_pass=args.require_full_pass,
        max_length=args.max_length,
        static=(args.augment == "False"),
        seed=args.seed,
    )
    val_ds = build_sft_dataset(
        merged_path=args.val_merged, tokenizer=tokenizer,
        top_p=args.top_p, sort_by=args.sort_by,
        require_full_pass=args.require_full_pass,
        max_length=args.max_length,
        static=True,   # val 始终静态,保证 eval_loss 可比
        seed=args.seed,
    )

    # ---- collator ----
    resp_ids = response_template_ids(tokenizer)
    collator = DynamicSFTCollator(response_template=resp_ids, tokenizer=tokenizer)

    # ---- trl config ----
    cfg = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True, fp16=False,                    # A800: bf16(StarCoder2 原生)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,              # 保留 candidate_* 字段
        max_seq_length=args.max_length,
        dataset_kwargs={"skip_prepare_dataset": True},   # 跳过 trl 自动 tokenization
        report_to=["tensorboard"],
        deepspeed=args.ds_config,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,   # trl 0.12+ 用 processing_class 替代 tokenizer 参数
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        trainer.train()

    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
