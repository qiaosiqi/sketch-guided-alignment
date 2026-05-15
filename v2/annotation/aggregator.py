"""
评分汇总。

主入口:
    score_one(judge, problem, sketch, code, alpha=0.4) -> AlgorithmScores | None

策略:
- 串行调两次 judge:先 sketch、再 code
- 任一失败,返回 AlgorithmScores(parsable=False, final=-1)
- final = alpha * mean(S1..S4) + (1 - alpha) * mean(C1..C5)
- 只对 pass_ratio >= pass_threshold 的解评分(过滤接口在 batch_annotate)
"""
from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Optional

from .judge_client import GLMJudge
from .rubric import score_sketch, score_code
from ..data.schema import (
    Problem, SketchScores, CodeScores, AlgorithmScores,
)


log = logging.getLogger(__name__)


def _mean(*xs) -> float:
    return sum(xs) / len(xs)


def aggregate_final(s: SketchScores, c: CodeScores, alpha: float = 0.4) -> float:
    sketch_mean = _mean(s.S1_correctness, s.S2_specificity, s.S3_complexity_awareness, s.S4_edge_coverage)
    code_mean = _mean(c.C1_faithfulness, c.C2_time_complexity, c.C3_space_complexity,
                      c.C4_readability, c.C5_edge_handling)
    return alpha * sketch_mean + (1 - alpha) * code_mean


def score_one(
    judge: GLMJudge,
    problem_text: str,
    sketch: str,
    code: str,
    alpha: float = 0.4,
) -> AlgorithmScores:
    """评一份 (sketch, code)。无论成败都返回 AlgorithmScores。"""
    s = score_sketch(judge, problem_text, sketch)
    if s is None:
        return _failed("GLM-4-Air")
    c = score_code(judge, problem_text, sketch, code)
    if c is None:
        return _failed("GLM-4-Air")
    final = aggregate_final(s, c, alpha=alpha)
    return AlgorithmScores(sketch=s, code=c, final=final, judge_model="GLM-4-Air", parsable=True)


def _failed(judge_model: str) -> AlgorithmScores:
    return AlgorithmScores(
        sketch=SketchScores(0, 0, 0, 0, raw_response=""),
        code=CodeScores(0, 0, 0, 0, 0, raw_response=""),
        final=-1.0,
        judge_model=judge_model,
        parsable=False,
    )


# ============================================================
# 批处理:读 codes.jsonl + exec.jsonl,只对 pass_ratio>=θ 的解评分
# 增量写 scores.jsonl
# ============================================================

def batch_annotate(
    judge: GLMJudge,
    codes_path: str,
    exec_path: str,
    problems_by_id: dict[str, Problem],
    out_path: str,
    pass_threshold: float = 0.0,
    alpha: float = 0.4,
    sleep_between: float = 0.0,
):
    """
    对 codes.jsonl 里每条 (parsed_ok=True) 的 code 调 judge 评分,落 scores.jsonl。

    过滤:
        - parsed_ok=False 的 code 跳过(但写一条 final=-1 的占位)
        - pass_ratio < pass_threshold 的也跳过评分(写 final=-1 占位)
        默认 pass_threshold=0,意味着只要 code 跑得起来就评分。
        若想节约 API,把 pass_threshold 设为 0.5(与 θ_pass_gvb 对齐:
        GvB 候选恰好全覆盖,且不多评一份)。
    """
    # 读 exec 结果,索引 (task_id, sample_id, code_id) -> pass_ratio
    exec_idx: dict[tuple, float] = {}
    with open(exec_path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            key = (e["task_id"], e["sample_id"], e.get("code_id", 0))
            exec_idx[key] = e["pass_ratio"]

    # 已完成的(用于断点续跑)
    done: set[tuple] = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    done.add((o["task_id"], o["sample_id"], o.get("code_id", 0)))
                except Exception:
                    pass
    log.info(f"annotate: {len(done)} already scored.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(codes_path, "r", encoding="utf-8") as fin, open(out_path, "a", encoding="utf-8") as fout:
        for line in fin:
            c = json.loads(line)
            key = (c["task_id"], c["sample_id"], c.get("code_id", 0))
            if key in done:
                continue

            # 不可评分的占位
            placeholder = None
            if not c.get("parsed_ok"):
                placeholder = "unparsable_code"
            elif exec_idx.get(key, 0.0) < pass_threshold:
                placeholder = "below_pass_threshold"

            if placeholder is not None:
                rec = {
                    "task_id": c["task_id"],
                    "sample_id": c["sample_id"],
                    "code_id": c.get("code_id", 0),
                    "placeholder": placeholder,
                    "scores": asdict(_failed(judge.model)),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            problem = problems_by_id.get(c["task_id"])
            if problem is None:
                continue

            scores = score_one(judge, problem.question, c["sketch"], c["code"], alpha=alpha)
            rec = {
                "task_id": c["task_id"],
                "sample_id": c["sample_id"],
                "code_id": c.get("code_id", 0),
                "placeholder": None,
                "scores": asdict(scores),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(key)
            if sleep_between > 0:
                time.sleep(sleep_between)
