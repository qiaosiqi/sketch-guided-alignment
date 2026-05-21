"""
DPO 训练入口。基于 trl 0.11+ 的 DPOTrainer / DPOConfig。

关键设计:
- ref_model 默认与 policy 同一份(trl 内部会 deepcopy 或 reuse 加 PEFT adapter)
- 5090 ×2 bf16:policy + ref 各 6GB 参数。ZeRO-2 不切参,2×32GB 上每卡要
  同时持有 12GB 参数 + 6GB 梯度切片(切了)+ 12GB optim 切片(切了)+ 激活,
  非常临界 → 默认走 ZeRO-3 (ds_zero3_2gpu.json) 把参数也切到 2 卡。
  极端 OOM 再 --use_lora(LoRA 模式下 ref 复用 policy 关闭 adapter,显存最省)
- 动态采样靠 dpo_dataset.py 的 set_transform 实现
"""
from __future__ import annotations
import argparse
import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOTrainer, DPOConfig

from .pair_builder import PairThresholds
from .dpo_dataset import build_dpo_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_merged", required=True)
    ap.add_argument("--val_merged", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--task", required=True, choices=["pvf", "qvs", "gvb", "all"])
    ap.add_argument("--augment", default="True", choices=["True", "False"],
                    help="True=动态采样 pair;False=静态")

    # pair 阈值(PvF / QvS 不用,只对 GvB 生效)
    ap.add_argument("--theta_pass_gvb", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=6.0)

    # DPO 训练超参
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--learning_rate", type=float, default=5e-7)
    ap.add_argument("--num_train_epochs", type=int, default=10)
    # 5090 ×2:per_device=2, grad-accum=4, 2 卡 → 有效 batch = 16(2×4×2),与 A800 等效
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--ds_config", default=None)

    # 可选:LoRA(若全参数 OOM)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

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
    tokenizer.padding_side = "left"   # DPO 通常 left pad,decoder 友好

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.config.use_cache = False

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules="all-linear",   # 适配大多数 decoder-only
        )

    # ---- datasets ----
    th = PairThresholds(
        theta_pass_gvb=args.theta_pass_gvb, tau=args.tau,
    )
    train_ds = build_dpo_dataset(
        merged_path=args.train_merged, task=args.task, thresholds=th,
        static=(args.augment == "False"), seed=args.seed,
    )
    val_ds = build_dpo_dataset(
        merged_path=args.val_merged, task=args.task, thresholds=th,
        static=True,   # val 始终静态
        seed=args.seed,
    )

    # ---- trl config ----
    cfg = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
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
        bf16=True, fp16=False,                    # Blackwell + StarCoder2 都原生 bf16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        report_to=["tensorboard"],
        deepspeed=args.ds_config,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None if peft_config is not None else model.name_or_path,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        trainer.train()

    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
