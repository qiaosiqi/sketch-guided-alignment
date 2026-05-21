"""
codes.jsonl 并发执行入口(供 02_sample_pilot / eval_sampling 共用)。

每个 task 调用 run_one_solution,而 run_one_solution 内部已经用
multiprocessing.Process 把用户代码隔离到子进程 —— 所以这里用线程池就够:
线程在子进程 join 时让出 GIL,不会互相阻塞,而且不用把 Problem 序列化到
worker、也避免 ProcessPoolExecutor → multiprocessing.Process 的 fork-of-fork。

主线程独占 out_path 的 append 写入,无文件锁问题。
"""
from __future__ import annotations
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from ..data.schema import Problem, ExecutionResult
from .runner import run_one_solution


log = logging.getLogger(__name__)


def default_exec_workers() -> int:
    """默认 worker 数:留 4 核给 vllm/系统/对端 shard,其余对半分。

    dp_sample 双 shard 同时跑 execution 阶段时,两个进程共享 CPU,
    所以 28 核 → 每 shard 12 worker 较稳。可用 --exec_workers 显式覆盖,
    也可设环境变量 V2_N_SHARDS 改预设的 shard 数。
    """
    cpu = os.cpu_count() or 4
    n_shards = int(os.environ.get("V2_N_SHARDS", "2"))
    return max(1, (cpu - 4) // max(1, n_shards))


def run_executions_parallel(
    codes_path: str,
    problems_by_id: dict[str, Problem],
    out_path: str,
    do_timing: bool = False,
    timeout_per_test: float = 3.0,
    max_workers: int | None = None,
    log_every: int = 100,
) -> None:
    """并发执行 codes_path 里所有 code,append 写 out_path。

    - 断点续跑:跳过 out_path 中已存在的 (task_id, sample_id, code_id)
    - parsed_ok=False 的 code 直接写一条 0 分占位,不进 worker
    - 顺序:exec.jsonl 不保证与 codes.jsonl 同序(downstream 按 key 索引,无影响)
    """
    if max_workers is None:
        max_workers = default_exec_workers()
    log.info(
        f"run_executions_parallel: max_workers={max_workers} do_timing={do_timing}"
    )

    done_keys: set = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_keys.add(
                        (obj["task_id"], obj["sample_id"], obj.get("code_id", 0))
                    )
                except Exception:
                    pass

    placeholder_rows: list[dict] = []
    runnable: list[dict] = []
    with open(codes_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            key = (c["task_id"], c["sample_id"], c.get("code_id", 0))
            if key in done_keys:
                continue
            if not c.get("parsed_ok"):
                er = ExecutionResult(
                    task_id=c["task_id"], sample_id=c["sample_id"],
                    code_id=c.get("code_id", 0),
                    n_tests=0, n_passed=0, pass_ratio=0.0,
                    per_test_pass=[], error="unparsable_code",
                )
                placeholder_rows.append(asdict(er))
                continue
            if c["task_id"] not in problems_by_id:
                continue
            runnable.append(c)

    def _task(c: dict) -> ExecutionResult:
        problem = problems_by_id[c["task_id"]]
        try:
            return run_one_solution(
                problem, c["code"],
                timeout_per_test=timeout_per_test,
                do_timing=do_timing,
                sample_id=c["sample_id"],
                code_id=c.get("code_id", 0),
            )
        except Exception as e:
            return ExecutionResult(
                task_id=c["task_id"], sample_id=c["sample_id"],
                code_id=c.get("code_id", 0),
                n_tests=len(problem.inputs), n_passed=0, pass_ratio=0.0,
                per_test_pass=[False] * len(problem.inputs),
                error=f"runner_crash: {e}",
            )

    with open(out_path, "a", encoding="utf-8") as fout:
        for row in placeholder_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()

        if not runnable:
            log.info(f"nothing to execute (placeholders written: {len(placeholder_rows)})")
            return

        done_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_task, c) for c in runnable]
            for fut in as_completed(futures):
                er = fut.result()
                fout.write(json.dumps(asdict(er), ensure_ascii=False) + "\n")
                fout.flush()
                done_count += 1
                if done_count % log_every == 0:
                    log.info(f"executed {done_count}/{len(runnable)} codes")
        log.info(
            f"executed {done_count}/{len(runnable)} codes "
            f"(+ {len(placeholder_rows)} unparsable placeholders)"
        )
