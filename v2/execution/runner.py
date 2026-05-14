"""
统一执行入口。

run_one_solution(problem, code, timeout_per_test=3.0, time_total=False)
    → ExecutionResult

设计要点:
- 子进程隔离:每个 (code, problem) 起一个 multiprocessing.Process,跑完所有测试用例再回收
- partial credit:逐 test case 标记 pass/fail,最终 pass_ratio = n_passed / n_tests
- 测时:仅当 pass_ratio == 1.0 时,对"整套测试套件作为一次"再跑 stable_runtime
- timeout:每个 test case 单独超时(time_limit),硬墙(子进程 join timeout)是 n_tests * timeout + 5
"""
from __future__ import annotations
import multiprocessing
import sys
import traceback
from typing import Any

from ..data.schema import Problem, ExecutionResult
from .sandbox import (
    time_limit, swallow_io, create_tempdir, reliability_guard,
)
from .timing import stable_runtime
from .compare import compare_stdio, compare_fncall


# ============================================================
# 子进程入口
# ============================================================

def _worker(
    code: str,
    io_format: str,
    fn_name: str | None,
    inputs: list,
    outputs: list,
    timeout_per_test: float,
    do_timing: bool,
    result_pipe,
):
    """在子进程内执行用户代码并比对所有测试用例。结果写入 result_pipe。"""
    per_test = []
    error_str: str | None = None
    timed_out_any = False
    timing_mean: float | None = None
    timing_std: float | None = None

    try:
        with create_tempdir():
            reliability_guard()

            # 先 compile 整段代码,任何语法错都在这里炸
            try:
                code_obj = compile(code, "<solution>", "exec")
            except BaseException as e:
                error_str = f"compile_error: {type(e).__name__}: {e}"
                # 全部记为 fail
                per_test = [False] * len(inputs)
                result_pipe.send({
                    "per_test_pass": per_test, "error": error_str,
                    "timed_out": True, "timing_mean": None, "timing_std": None,
                })
                return

            # 函数式题需要先 exec 一次拿到函数对象,后续直接调用
            persistent_globals: dict[str, Any] = {}
            if io_format == "fncall":
                try:
                    with swallow_io():
                        with time_limit(timeout_per_test):
                            exec(code_obj, persistent_globals)
                except BaseException as e:
                    error_str = f"top_level_error: {type(e).__name__}: {e}"
                    per_test = [False] * len(inputs)
                    result_pipe.send({
                        "per_test_pass": per_test, "error": error_str,
                        "timed_out": isinstance(e, TimeoutError),
                        "timing_mean": None, "timing_std": None,
                    })
                    return
                fn = persistent_globals.get(fn_name or "")
                if not callable(fn):
                    per_test = [False] * len(inputs)
                    error_str = f"function_not_found: {fn_name}"
                    result_pipe.send({
                        "per_test_pass": per_test, "error": error_str,
                        "timed_out": False, "timing_mean": None, "timing_std": None,
                    })
                    return

            # 逐 test case 跑
            for idx, (inp, exp) in enumerate(zip(inputs, outputs)):
                try:
                    if io_format == "stdio":
                        stdin_str = _build_stdin(inp)
                        with swallow_io(stdin_str=stdin_str) as holder:
                            with time_limit(timeout_per_test):
                                exec(code_obj, {"__name__": "__main__"})
                        ok = compare_stdio(holder.stdout, exp)
                    else:  # fncall
                        args = _build_fn_args(inp)
                        with swallow_io():
                            with time_limit(timeout_per_test):
                                ret = fn(*args)
                        ok = compare_fncall(ret, exp)
                    per_test.append(bool(ok))
                except TimeoutError:
                    per_test.append(False)
                    timed_out_any = True
                except BaseException:
                    per_test.append(False)

            # 测时:仅当全过时才测
            if do_timing and all(per_test):
                def _runall():
                    if io_format == "stdio":
                        for inp in inputs:
                            stdin_str = _build_stdin(inp)
                            with swallow_io(stdin_str=stdin_str):
                                exec(code_obj, {"__name__": "__main__"})
                    else:
                        for inp in inputs:
                            args = _build_fn_args(inp)
                            with swallow_io():
                                fn(*args)
                try:
                    with time_limit(timeout_per_test * len(inputs) * 2):
                        timing_mean, timing_std = stable_runtime(_runall)
                except TimeoutError:
                    timing_mean, timing_std = None, None

    except BaseException as e:
        error_str = f"worker_crashed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        if not per_test:
            per_test = [False] * len(inputs)

    result_pipe.send({
        "per_test_pass": per_test,
        "error": error_str,
        "timed_out": timed_out_any,
        "timing_mean": timing_mean,
        "timing_std": timing_std,
    })


# ============================================================
# stdin / args 构造
# ============================================================

def _build_stdin(test_input: Any) -> str:
    """APPS stdio 题:输入要么是 list[str](多行),要么是 str。"""
    if isinstance(test_input, list):
        return "\n".join(str(x) for x in test_input) + "\n"
    return str(test_input) + ("" if str(test_input).endswith("\n") else "\n")


def _build_fn_args(test_input: Any) -> list:
    """APPS fncall 题:输入是 list[args],对 fn(*args) 调用。"""
    if isinstance(test_input, list):
        return list(test_input)
    return [test_input]


# ============================================================
# 公开入口
# ============================================================

def run_one_solution(
    problem: Problem,
    code: str,
    timeout_per_test: float = 3.0,
    do_timing: bool = True,
    sample_id: int = 0,
    code_id: int = 0,
) -> ExecutionResult:
    """执行一份候选代码,返回 ExecutionResult。完全隔离在子进程内。"""
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(
        target=_worker,
        args=(
            code, problem.io_format, problem.fn_name,
            problem.inputs, problem.outputs,
            timeout_per_test, do_timing, child_conn,
        ),
    )
    p.start()
    hard_wall = timeout_per_test * len(problem.inputs) * (60 if do_timing else 2) + 10
    p.join(timeout=hard_wall)
    if p.is_alive():
        p.kill()
        p.join()
        return ExecutionResult(
            task_id=problem.task_id, sample_id=sample_id, code_id=code_id,
            n_tests=len(problem.inputs), n_passed=0, pass_ratio=0.0,
            per_test_pass=[False] * len(problem.inputs),
            error="hard_wall_timeout", timed_out=True,
        )

    if not parent_conn.poll():
        return ExecutionResult(
            task_id=problem.task_id, sample_id=sample_id, code_id=code_id,
            n_tests=len(problem.inputs), n_passed=0, pass_ratio=0.0,
            per_test_pass=[False] * len(problem.inputs),
            error="no_result", timed_out=True,
        )

    msg = parent_conn.recv()
    per_test = msg["per_test_pass"]
    n_tests = len(problem.inputs)
    n_passed = sum(1 for x in per_test if x)
    return ExecutionResult(
        task_id=problem.task_id,
        sample_id=sample_id,
        code_id=code_id,
        n_tests=n_tests,
        n_passed=n_passed,
        pass_ratio=(n_passed / n_tests) if n_tests else 0.0,
        per_test_pass=list(per_test),
        error=msg.get("error"),
        runtime_ns_mean=msg.get("timing_mean"),
        runtime_ns_std=msg.get("timing_std"),
        timed_out=bool(msg.get("timed_out")),
    )
