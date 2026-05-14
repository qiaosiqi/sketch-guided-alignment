"""
把 sketches.jsonl + codes.jsonl + exec.jsonl + scores.jsonl 合并成 merged.jsonl。
每行 = 一题,answers 数组里是该题所有候选解(过滤后)。

输入(逐条):
    sketches.jsonl: SketchSample
    codes.jsonl:    CodeSample (1 sketch:1 code)
    exec.jsonl:     ExecutionResult
    scores.jsonl:   {task_id, sample_id, code_id, placeholder, scores}

输出:
    merged.jsonl:   MergedProblem (含 difficulty/io_format,answers: List[MergedAnswer])

题目级过滤:
    - 题没有任何 parsed_ok=True 的 code → 整题丢
    - answers 内已经按 code_id 排序

用法:
    python -m v2.merge.build_dataset \
        --problems_jsonl out/apps/train.jsonl \
        --sample_dir out/main \
        --out out/datasets/train/merged.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Optional

from ..data.schema import (
    Problem, MergedAnswer, MergedProblem,
)


def _index_jsonl(path: str, key_fn) -> dict:
    """读 jsonl,返回 {key: row} 字典。"""
    idx = {}
    if not os.path.exists(path):
        return idx
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx[key_fn(o)] = o
    return idx


def load_problems(path: str) -> dict[str, Problem]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            p = Problem(**o)
            out[p.task_id] = p
    return out


def build_merged(
    problems_by_id: dict[str, Problem],
    sample_dir: str,
    out_path: str,
):
    codes_path = os.path.join(sample_dir, "codes.jsonl")
    exec_path = os.path.join(sample_dir, "exec.jsonl")
    scores_path = os.path.join(sample_dir, "scores.jsonl")

    exec_idx = _index_jsonl(exec_path, lambda o: (o["task_id"], o["sample_id"], o.get("code_id", 0)))
    score_idx = _index_jsonl(scores_path, lambda o: (o["task_id"], o["sample_id"], o.get("code_id", 0)))

    # 按 task_id 分组 codes
    by_task: dict[str, list[dict]] = defaultdict(list)
    with open(codes_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            by_task[c["task_id"]].append(c)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_total, n_kept = 0, 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for task_id, codes in by_task.items():
            problem = problems_by_id.get(task_id)
            if problem is None:
                continue
            n_total += 1

            answers: list[MergedAnswer] = []
            for c in sorted(codes, key=lambda x: (x["sample_id"], x.get("code_id", 0))):
                if not c.get("parsed_ok"):
                    continue
                key = (c["task_id"], c["sample_id"], c.get("code_id", 0))
                e = exec_idx.get(key)
                if e is None:
                    continue

                s = score_idx.get(key)
                algo_final = -1.0
                algo_breakdown: Optional[dict] = None
                if s is not None and s.get("scores") and s["scores"].get("parsable"):
                    algo_final = s["scores"]["final"]
                    algo_breakdown = {
                        "sketch": {
                            k: v for k, v in s["scores"]["sketch"].items()
                            if k != "raw_response"
                        },
                        "code": {
                            k: v for k, v in s["scores"]["code"].items()
                            if k != "raw_response"
                        },
                    }

                answers.append(MergedAnswer(
                    sketch=c["sketch"],
                    code=c["code"],
                    pass_ratio=e["pass_ratio"],
                    n_tests=e["n_tests"],
                    runtime_ns_mean=e.get("runtime_ns_mean"),
                    runtime_ns_std=e.get("runtime_ns_std"),
                    algo_final=algo_final,
                    algo_breakdown=algo_breakdown,
                    sample_temp=c.get("sample_temp", 0.6),
                ))

            if not answers:
                continue

            mp = MergedProblem(
                task_id=problem.task_id,
                difficulty=problem.difficulty,
                io_format=problem.io_format,
                question=problem.question,
                fn_name=problem.fn_name,
                answers=[asdict(a) for a in answers],
            )
            fout.write(json.dumps(asdict(mp), ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"merged: kept {n_kept}/{n_total} problems")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems_jsonl", required=True)
    ap.add_argument("--sample_dir", required=True, help="目录含 codes.jsonl exec.jsonl scores.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    problems = load_problems(args.problems_jsonl)
    build_merged(problems, args.sample_dir, args.out)


if __name__ == "__main__":
    main()
