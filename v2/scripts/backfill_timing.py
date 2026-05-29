"""
对已有 eval 目录回填 timing:复用 codes.jsonl,重跑 execution 阶段开 --do_timing。

用法:
    python -m v2.scripts.backfill_timing \
        --eval_dir /root/shared-nvme/work/out/evals/base \
        --problems_jsonl /root/shared-nvme/work/out/apps/test.jsonl \
        --exec_workers 24

注意:执行前请确保已经把原 exec.jsonl 改名/备份(否则 run_executions_parallel
的断点续跑逻辑会 skip 全部条目什么也不做)。
"""
from __future__ import annotations
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging

from ..data.schema import Problem
from ..execution import run_executions_parallel


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("backfill_timing")


def load_problems(path: str) -> dict[str, Problem]:
    out: dict[str, Problem] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = Problem(**json.loads(line))
            out[p.task_id] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="目录里需有 codes.jsonl")
    ap.add_argument("--problems_jsonl", required=True)
    ap.add_argument("--exec_workers", type=int, default=24)
    ap.add_argument("--timeout_per_test", type=float, default=3.0)
    args = ap.parse_args()

    codes_path = os.path.join(args.eval_dir, "codes.jsonl")
    exec_path = os.path.join(args.eval_dir, "exec.jsonl")
    assert os.path.exists(codes_path), f"missing {codes_path}"
    if os.path.exists(exec_path):
        log.warning(
            f"{exec_path} exists — run_executions_parallel 会 dedup,可能 skip 全部条目。"
            "如要完整回填,请先 mv exec.jsonl exec.jsonl.bak"
        )

    problems_by_id = load_problems(args.problems_jsonl)
    log.info(f"loaded {len(problems_by_id)} problems; eval_dir={args.eval_dir}")

    # 单进程 execution(没有 dp_eval 的 sharding,因为不再起 vllm 占卡)
    # 128 vCPU 富余,exec_workers=24 留出大量空间给 timing 稳定性
    run_executions_parallel(
        codes_path=codes_path,
        problems_by_id=problems_by_id,
        out_path=exec_path,
        do_timing=True,
        timeout_per_test=args.timeout_per_test,
        max_workers=args.exec_workers,
    )
    log.info("backfill done")


if __name__ == "__main__":
    main()
