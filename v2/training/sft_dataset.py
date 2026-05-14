"""
SFT 数据集构造:从 merged.jsonl → HF Dataset。

每行(题)在 dataset 里仍然是一行,但 "candidates" 字段是该题 top-p% 答案的预 tokenize 列表。
训练时由 collator 每步随机挑一份(Dynamic Solution Selection)。

`build_sft_dataset(merged_path, tokenizer, top_p, sort_by, max_length, static)`:
    sort_by ∈ {"algo_final", "pass_ratio", "runtime"}
    static=True  → 每题只保留 1 个候选(随机一次,后续训练每个 epoch 都用它)
    static=False → 每题保留 top-p% 全部,collator 每步随机抽
"""
from __future__ import annotations
import json
import math
import random
from typing import Literal

from datasets import Dataset, disable_caching

from ..data.prompts import build_training_text, RESPONSE_TEMPLATE


disable_caching()  # 重要:动态选择不能让 HF cache 把候选集 freeze 住


SortKey = Literal["algo_final", "pass_ratio", "runtime"]


def _sort_value(ans: dict, sort_by: SortKey) -> float:
    """返回排序键,越小越好(top-p% 取小的)。"""
    if sort_by == "algo_final":
        v = ans.get("algo_final", -1.0)
        # 没评分的 (-1.0) 排到最后
        return -v if v >= 0 else float("inf")
    if sort_by == "pass_ratio":
        v = ans.get("pass_ratio", 0.0)
        return -v   # pass_ratio 高 = 好,所以取负
    if sort_by == "runtime":
        v = ans.get("runtime_ns_mean")
        if v is None:
            return float("inf")
        return float(v)
    raise ValueError(f"unknown sort_by: {sort_by}")


def _select_topp(answers: list[dict], top_p: int, sort_by: SortKey,
                 require_full_pass: bool = False) -> list[dict]:
    """按 sort_by 升序,取 top-p% (向上取整)。可选只保留 pass_ratio==1.0 的(QvS 友好)。"""
    pool = answers
    if require_full_pass:
        pool = [a for a in pool if a.get("pass_ratio", 0.0) == 1.0]
    if not pool:
        return []
    pool = sorted(pool, key=lambda a: _sort_value(a, sort_by))
    k = max(1, math.ceil(top_p * len(pool) / 100))
    return pool[:k]


def build_sft_dataset(
    merged_path: str,
    tokenizer,
    top_p: int = 25,
    sort_by: SortKey = "algo_final",
    require_full_pass: bool = False,
    max_length: int = 2048,
    static: bool = False,
    seed: int = 1,
) -> Dataset:
    """读 merged.jsonl,过滤后产出 HF Dataset。每行:
        {
            "task_id": str,
            "candidate_input_ids": List[List[int]],
            "candidate_attention_mask": List[List[int]],
        }
    """
    rng = random.Random(seed)
    rows = []
    skipped_empty = 0

    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            problem = json.loads(line)
            top = _select_topp(problem["answers"], top_p, sort_by, require_full_pass)
            if not top:
                skipped_empty += 1
                continue
            if static:
                top = [rng.choice(top)]

            input_ids_list = []
            attn_list = []
            for ans in top:
                text = build_training_text(
                    question=problem["question"],
                    sketch=ans["sketch"],
                    code=ans["code"],
                )
                enc = tokenizer(
                    text, truncation=True, padding=False, max_length=max_length,
                    add_special_tokens=True,
                )
                input_ids_list.append(enc["input_ids"])
                attn_list.append(enc["attention_mask"])

            rows.append({
                "task_id": problem["task_id"],
                "candidate_input_ids": input_ids_list,
                "candidate_attention_mask": attn_list,
            })

    print(f"SFT dataset: kept {len(rows)} problems, skipped {skipped_empty} empty (no top-p%) problems")
    return Dataset.from_list(rows)


def response_template_ids(tokenizer) -> list[int]:
    """tokenize RESPONSE_TEMPLATE,用于 DataCollatorForCompletionOnlyLM 切 label。
    某些 tokenizer 在串首加 BOS,要剥掉以便子串匹配。"""
    ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    return ids
