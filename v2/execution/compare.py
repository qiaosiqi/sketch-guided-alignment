"""
APPS 测试用例的输出比对逻辑。

APPS 的 `outputs` 字段格式很乱:
- stdio:每个 test case 的期望输出可能是单个字符串、字符串列表(多行)、嵌套列表
- fncall:期望是函数返回值,可能是 int/float/str/list/tuple/dict/bool

比对策略:
- stdio:规范化(strip 行尾空白、忽略空行)后字符串比较;允许浮点 ε 容差
- fncall:递归比较;数字用 isclose,容器递归,其余 ==

参考 APPS 官方 evaluator 实现的常见处理。
"""
from __future__ import annotations
import math
from typing import Any


FLOAT_EPS = 1e-6


def _normalize_stdio_lines(text: str) -> list[str]:
    """stdio 输出规范化为去尾空行的行列表。"""
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    return [ln.rstrip() for ln in lines]


def _expected_to_lines(expected: Any) -> list[str]:
    """把 APPS 期望输出统一成行列表。"""
    if isinstance(expected, list):
        out: list[str] = []
        for item in expected:
            if isinstance(item, list):
                out.extend(str(x) for x in item)
            else:
                out.extend(str(item).split("\n"))
        # 去尾空行
        while out and out[-1].strip() == "":
            out.pop()
        return [ln.rstrip() for ln in out]
    if isinstance(expected, str):
        return _normalize_stdio_lines(expected)
    return [str(expected)]


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def compare_stdio(actual_stdout: str, expected: Any) -> bool:
    """逐行比对 stdio 输出。允许浮点 ε 容差。"""
    actual_lines = _normalize_stdio_lines(actual_stdout)
    expected_lines = _expected_to_lines(expected)

    if len(actual_lines) != len(expected_lines):
        return False

    for a, e in zip(actual_lines, expected_lines):
        if a == e:
            continue
        # 浮点容差
        fa, fe = _try_float(a), _try_float(e)
        if fa is not None and fe is not None:
            if math.isclose(fa, fe, rel_tol=FLOAT_EPS, abs_tol=FLOAT_EPS):
                continue
        # 多 token 行:逐 token 比
        ta, te = a.split(), e.split()
        if len(ta) == len(te):
            ok = True
            for x, y in zip(ta, te):
                if x == y:
                    continue
                fx, fy = _try_float(x), _try_float(y)
                if fx is not None and fy is not None and math.isclose(fx, fy, rel_tol=FLOAT_EPS, abs_tol=FLOAT_EPS):
                    continue
                ok = False
                break
            if ok:
                continue
        return False
    return True


def compare_fncall(actual: Any, expected: Any) -> bool:
    """递归比较函数返回值与期望。APPS 期望通常是包了一层 list 的。"""
    # APPS 习惯把每个 test case 的 expected 包成 [v]。优先剥皮,再回退原值。
    if isinstance(expected, list) and len(expected) == 1:
        if _deep_equal(actual, expected[0]):
            return True
    return _deep_equal(actual, expected)


def _deep_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=FLOAT_EPS, abs_tol=FLOAT_EPS)
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_deep_equal(a[k], b[k]) for k in a)
    return a == b
