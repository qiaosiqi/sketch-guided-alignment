"""
APPS 原始数据 → 统一 Problem schema。

APPS 目录结构(每个 split):
    raw/{split}/{0000..9999}/
        question.txt
        input_output.json   {"inputs": [...], "outputs": [...], 可选 "fn_name": ...}
        solutions.json
        metadata.json       {"difficulty": "...", "url": "..."}
        starter_code.py     (可选)

判定题型:
    input_output.json 含 "fn_name" → fncall;否则 stdio。
    (注:有 starter_code.py 但 input_output 没 fn_name 的极少数情况,按 stdio 处理。)

过滤逻辑:
    - 只保留 difficulty in {"interview", "competition"}
    - 跳过解析失败 / 无测试用例 的题
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Iterable

from .schema import Problem


VALID_DIFFICULTIES = ("interview", "competition")


def _read_text(p: Path) -> str | None:
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def load_one(problem_dir: Path, split: str) -> Problem | None:
    """从一个题目目录读出 Problem。读不出来或不符合条件返回 None。"""
    meta_p = problem_dir / "metadata.json"
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    difficulty = meta.get("difficulty")
    if difficulty not in VALID_DIFFICULTIES:
        return None

    io_p = problem_dir / "input_output.json"
    if not io_p.exists():
        return None
    try:
        io = json.loads(io_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    if not inputs or not outputs or len(inputs) != len(outputs):
        return None

    fn_name = io.get("fn_name")
    io_format = "fncall" if fn_name else "stdio"

    question = _read_text(problem_dir / "question.txt")
    if not question:
        return None

    starter = _read_text(problem_dir / "starter_code.py")

    return Problem(
        task_id=f"apps_{split}_{problem_dir.name}",
        difficulty=difficulty,
        io_format=io_format,
        question=question,
        inputs=inputs,
        outputs=outputs,
        fn_name=fn_name,
        starter_code=starter,
        url=meta.get("url"),
    )


def iter_split(apps_root: str | Path, split: str) -> Iterable[Problem]:
    """遍历一个 split 下所有题。`split` 为 'train' 或 'test'。"""
    root = Path(apps_root) / split
    if not root.exists():
        raise FileNotFoundError(f"APPS split directory missing: {root}")
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        p = load_one(sub, split)
        if p is not None:
            yield p


def load_train_val(
    apps_root: str | Path,
    val_ratio: float = 0.1,
    seed: int = 1,
) -> tuple[list[Problem], list[Problem]]:
    """从 APPS train 加载并过滤后,9:1 随机拆 train / val。固定 seed。"""
    all_probs = list(iter_split(apps_root, "train"))
    rng = random.Random(seed)
    rng.shuffle(all_probs)
    n_val = int(len(all_probs) * val_ratio)
    val = all_probs[:n_val]
    train = all_probs[n_val:]
    return train, val


def load_test(apps_root: str | Path) -> list[Problem]:
    """APPS 官方 test 集(过滤后)全量。"""
    return list(iter_split(apps_root, "test"))


# ============================================================
# CLI: 把过滤+拆分结果落盘成 jsonl,供下游消费
# ============================================================

def main():
    import argparse
    from dataclasses import asdict

    ap = argparse.ArgumentParser()
    ap.add_argument("--apps_root", required=True, help="APPS raw/ 上一级目录")
    ap.add_argument("--out_dir", required=True, help="输出目录,会产生 train.jsonl / val.jsonl / test.jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train, val = load_train_val(args.apps_root, args.val_ratio, args.seed)
    test = load_test(args.apps_root)

    def dump(probs: list[Problem], name: str):
        p = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for prob in probs:
                f.write(json.dumps(asdict(prob), ensure_ascii=False) + "\n")
        diffs = {}
        for prob in probs:
            diffs[prob.difficulty] = diffs.get(prob.difficulty, 0) + 1
        fmts = {}
        for prob in probs:
            fmts[prob.io_format] = fmts.get(prob.io_format, 0) + 1
        print(f"[{name}] total={len(probs)}  difficulty={diffs}  io_format={fmts}")

    dump(train, "train")
    dump(val, "val")
    dump(test, "test")


if __name__ == "__main__":
    main()
