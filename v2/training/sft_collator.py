"""
SFT collator:动态从每题候选池里随机挑一个 sample,然后走 trl 的
DataCollatorForCompletionOnlyLM 做 completion-only label 屏蔽。

Loss 仅在 RESPONSE_TEMPLATE 之后开始计算(即模型从 problem 推 sketch+code 那段)。
"""
from __future__ import annotations
import random
from typing import Any, Dict, List

from trl import DataCollatorForCompletionOnlyLM


class DynamicSFTCollator(DataCollatorForCompletionOnlyLM):
    """
    输入 batch features:每行有 candidate_input_ids / candidate_attention_mask 两个 list。
    每步从每题的 candidates 里随机挑一份,再调父类拼 batch + 屏蔽 prompt 部分 label。
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        flat = []
        for f in features:
            cands_ids = f["candidate_input_ids"]
            cands_attn = f["candidate_attention_mask"]
            i = random.randrange(len(cands_ids))
            flat.append({
                "input_ids": cands_ids[i],
                "attention_mask": cands_attn[i],
            })
        return super().__call__(flat)
