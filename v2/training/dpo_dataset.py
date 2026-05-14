"""
DPO 数据集:每行存一题的全部候选,运行时用 set_transform 每步动态采一个 pair。

输出每行(经过 transform 后):
    {
        "prompt":   str (### Problem ... ### Sketch\n),
        "chosen":   str (sketch+code 拼接,即 response),
        "rejected": str
    }

trl DPOTrainer (0.11+) 接受 prompt/chosen/rejected 三个 string column,自己内部处理
tokenization 和 ref_logp 计算。

动态采样原理:HF Dataset.set_transform 在每次 __getitem__ 时调用,所以同一题
跨 epoch 会拿到不同的 (chosen, rejected) 对。
"""
from __future__ import annotations
import json
import random
from typing import Literal

from datasets import Dataset, disable_caching

from ..data.prompts import TRAINING_TEMPLATE
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
    static: bool = False,
    seed: int = 1,
) -> Dataset:
    """
    读 merged.jsonl,过滤可构造该 task 偏好对的题。
    static=False:set_transform 动态;static=True:生成一次性 (prompt,chosen,rejected) 列。
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

    ds = Dataset.from_list(rows)

    if static:
        rng = random.Random(seed)

        def _mat(row):
            pair = sample_pair(row["candidates"], task, thresholds, rng)
            chosen, rejected = pair  # type: ignore[misc]
            return {
                "prompt": _build_prompt_part(row["question"]),
                "chosen": _build_response_part(chosen["sketch"], chosen["code"]),
                "rejected": _build_response_part(rejected["sketch"], rejected["code"]),
            }

        ds = ds.map(_mat, remove_columns=["question", "candidates"])
        return ds

    # 动态:set_transform 让每次访问都重抽
    def _transform(batch):
        n = len(batch["question"])
        out_prompt = []
        out_chosen = []
        out_rejected = []
        for i in range(n):
            cands = batch["candidates"][i]
            q = batch["question"][i]
            pair = sample_pair(cands, task, thresholds)
            if pair is None:
                # 极少数情况(并发问题导致筛过的题在此处抽不出),退化为同份
                first = cands[0]
                out_prompt.append(_build_prompt_part(q))
                out_chosen.append(_build_response_part(first["sketch"], first["code"]))
                out_rejected.append(_build_response_part(first["sketch"], first["code"]))
                continue
            chosen, rejected = pair
            out_prompt.append(_build_prompt_part(q))
            out_chosen.append(_build_response_part(chosen["sketch"], chosen["code"]))
            out_rejected.append(_build_response_part(rejected["sketch"], rejected["code"]))
        return {
            "prompt": out_prompt,
            "chosen": out_chosen,
            "rejected": out_rejected,
        }

    ds.set_transform(_transform)
    return ds
