"""
execution/runner.py 的 smoke tests。

依赖 signal.SIGALRM,Windows 不支持,整文件 skip。
"""
import pytest

from v2.tests.conftest import IS_WINDOWS

pytestmark = pytest.mark.skipif(IS_WINDOWS, reason="runner uses signal.SIGALRM, Linux/macOS only")

from v2.execution.runner import run_one_solution


# ============================================================
# fncall
# ============================================================

def test_fncall_all_pass(fncall_problem):
    code = "def add(a, b):\n    return a + b\n"
    er = run_one_solution(fncall_problem, code, do_timing=False)
    assert er.n_tests == 3
    assert er.n_passed == 3
    assert er.pass_ratio == 1.0
    assert er.per_test_pass == [True, True, True]


def test_fncall_partial(fncall_problem):
    code = "def add(a, b):\n    return a + b if a > 0 else 0\n"
    er = run_one_solution(fncall_problem, code, do_timing=False)
    # 1+2=3 OK, 3+4=7 OK, 10+-5=5 OK 仍然全过 — 重新写一个真的部分失败的
    code2 = "def add(a, b):\n    return a - b\n"  # 都错
    er2 = run_one_solution(fncall_problem, code2, do_timing=False)
    assert er2.pass_ratio == 0.0


def test_fncall_partial_real_partial(fncall_problem):
    # 只对正数对正确
    code = "def add(a, b):\n    if b < 0: return 0\n    return a + b\n"
    er = run_one_solution(fncall_problem, code, do_timing=False)
    # cases: [1,2]→3 ok, [3,4]→7 ok, [10,-5]→0 错(期望 5)
    assert er.n_passed == 2
    assert abs(er.pass_ratio - 2/3) < 1e-6


def test_fncall_syntax_error(fncall_problem):
    code = "def add(a, b)\n    return a + b\n"   # 缺冒号
    er = run_one_solution(fncall_problem, code, do_timing=False)
    assert er.pass_ratio == 0.0
    assert er.error is not None
    assert "compile" in er.error or "syntax" in er.error.lower()


def test_fncall_runtime_error(fncall_problem):
    code = "def add(a, b):\n    return a / 0\n"
    er = run_one_solution(fncall_problem, code, do_timing=False)
    assert er.pass_ratio == 0.0


def test_fncall_missing_function(fncall_problem):
    code = "def subtract(a, b):\n    return a - b\n"
    er = run_one_solution(fncall_problem, code, do_timing=False)
    assert er.pass_ratio == 0.0


# ============================================================
# stdio
# ============================================================

def test_stdio_all_pass(stdio_problem):
    code = "n = int(input())\nprint(n + 1)\n"
    er = run_one_solution(stdio_problem, code, do_timing=False)
    assert er.n_tests == 3
    assert er.n_passed == 3


def test_stdio_partial(stdio_problem):
    # 只对正数加 1
    code = "n = int(input())\nprint(n + 1 if n > 0 else n)\n"
    er = run_one_solution(stdio_problem, code, do_timing=False)
    # cases: 5→6 ok, 100→101 ok, -3→-3 错(期望 -2)
    assert er.n_passed == 2


def test_stdio_timeout(stdio_problem):
    code = "n = int(input())\nwhile True: pass\n"
    er = run_one_solution(stdio_problem, code, do_timing=False, timeout_per_test=0.5)
    assert er.pass_ratio == 0.0
    assert er.timed_out
