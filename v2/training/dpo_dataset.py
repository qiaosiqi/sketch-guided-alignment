"""
DPO 数据集:每题预采 K 个 (chosen, rejected) pair,物化为 N×K 行的静态 Dataset。

每行输出:
    {
        "prompt":   str (### Problem ... ### Sketch\\n),
        "chosen":   str (sketch+code 拼接,即 response),
        "rejected": str
    }

设计选择(2026-05-21):
原版用 `set_transform` 做每步动态采样,但 trl 0.15 的 DPOTrainer 在 __init__ 时
就用 `dataset.map()` 预 tokenize,产出 prompt_input_ids 等列;set_transform 是 runtime hook,
会把 map 加上去的列影掉,导致 collator KeyError。

改为静态 K-pair 扩展:
- 预先对每题随机采 K 个 pair,展开成 N×K 行
- TRL 标准 map 路径正常工作
- K 由调用方控制;搭配 dpo_train.py 的"effective_epochs = E // K"联动,
  整体训练量(N × K × effective_E)与原 dynamic 模式(N × E)等价

trl DPOTrainer (0.11+) 接受 prompt/chosen/rejected 三个 string column,自己内部处理
tokenization 和 ref_logp 计算。
"""
from __future__ import annotations
import json
import random

from datasets import Dataset, disable_caching

from .pair_builder import (
    PairThresholds, Task, sample_pair, task_yieldable,
)


disable_caching()


def _build_prompt_part(question: str) -> str:
    """与 build_training_text 一致,但只到 '### Sketch\\n' 之前(即 response 的起点)。"""
    return f"### Problem\n{question.strip()}\n\n### Sketch\n"


def _build_response_part(sketch: str, code: str) -> str:
    """剩下的 response 段:sketch + Code fenced block。"""
    return f"{sketch.strip()}\n\n### Code\n```python\n{code.strip()}\n```"


def build_dpo_dataset(
    merged_path: str,
    task: Task,
    thresholds: PairThresholds,
    pairs_per_problem: int = 1,
    seed: int = 1,
) -> Dataset:
    """
    读 merged.jsonl,过滤可构造该 task 偏好对的题,每题预采 K=pairs_per_problem 个 pair。

    pairs_per_problem=1: 每题 1 个固定 pair(经典静态;val 集用)
    pairs_per_problem=K (K>1): 每题 K 个 pair(等效原 dynamic E=K;主跑 train 集用)

    采样 with replacement(`sample_pair` 内部用 `rng.choice`),候选少的题不会异常。
    seed 决定整次扩展的所有采样,同 seed 产出完全一致。
    """
    rows: list[dict] = []
    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if not task_yieldable(p["answers"], task, thresholds):
                continue
            rows.append({
                "task_id": p["task_id"],
                "question": p["question"],
                "candidates": p["answers"],
            })
    print(f"DPO[{task}]: kept {len(rows)} yieldable problems")

    K = max(1, pairs_per_problem)
    rng = random.Random(seed)

    expanded: list[dict] = []
    for row in rows:
        for _ in range(K):
            pair = sample_pair(row["candidates"], task, thresholds, rng)
            if pair is None:
                # 题级 task_yieldable 过滤已保证至少一对,理论上到不了
                continue
            chosen, rejected = pair
            expanded.append({
                "prompt": _build_prompt_part(row["question"]),
                "chosen": _build_response_part(chosen["sketch"], chosen["code"]),
                "rejected": _build_response_part(rejected["sketch"], rejected["code"]),
            })

    print(f"DPO[{task}]: expanded to {len(expanded)} rows ({len(rows)} problems × K={K})")
    return Dataset.from_list(expanded)
