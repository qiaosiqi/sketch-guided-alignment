"""execution/compare.py 的 smoke tests。"""
from v2.execution.compare import compare_stdio, compare_fncall


# ============================================================
# stdio
# ============================================================

def test_stdio_exact():
    assert compare_stdio("hello\nworld\n", ["hello", "world"])


def test_stdio_trailing_newlines():
    assert compare_stdio("hello\nworld\n\n\n", ["hello", "world"])


def test_stdio_mismatch_line_count():
    assert not compare_stdio("hello\n", ["hello", "world"])


def test_stdio_float_tolerance():
    assert compare_stdio("3.1415927\n", ["3.14159265"])


def test_stdio_token_level():
    # 多 token 行,可被空格分割逐项比较
    assert compare_stdio("1 2 3.0\n", ["1 2 3"])


def test_stdio_nested_list_expected():
    """APPS 风格:expected 是 list of lines。"""
    assert compare_stdio("a\nb\nc\n", [["a", "b", "c"]])


def test_stdio_single_string_expected():
    assert compare_stdio("yes\n", "yes")


# ============================================================
# fncall
# ============================================================

def test_fncall_int_wrapped():
    # APPS 风格期望值常包一层 list:[6]
    assert compare_fncall(6, [6])


def test_fncall_list_equal():
    assert compare_fncall([1, 2, 3], [[1, 2, 3]])


def test_fncall_float_close():
    assert compare_fncall(0.30000000004, [0.3])


def test_fncall_dict_equal():
    assert compare_fncall({"a": 1, "b": 2}, [{"a": 1, "b": 2}])


def test_fncall_mismatch():
    assert not compare_fncall(7, [6])


def test_fncall_nested():
    assert compare_fncall([[1, 2], [3]], [[[1, 2], [3]]])
