"""sampling/parser.py 的 smoke tests。"""
from v2.sampling.parser import parse_sketch, parse_code


# ============================================================
# parse_sketch
# ============================================================

def test_sketch_with_close_tag():
    s, ok = parse_sketch("Use DP with state f[i]. Time O(n).</SKETCH>\n### Code\n")
    assert ok
    assert s == "Use DP with state f[i]. Time O(n)."


def test_sketch_without_close_tag_terminates_at_header():
    s, ok = parse_sketch("Greedy approach over sorted intervals.\n### Code\n")
    assert ok
    assert "###" not in s
    assert "Greedy" in s


def test_sketch_with_leading_tag():
    s, ok = parse_sketch("<SKETCH>\nBinary search.</SKETCH>")
    assert ok
    assert s == "Binary search."


def test_sketch_empty():
    s, ok = parse_sketch("</SKETCH>")
    assert not ok
    assert s == ""


def test_sketch_too_short():
    s, ok = parse_sketch("hi</SKETCH>")
    assert not ok


def test_sketch_no_terminator_runs_to_end():
    s, ok = parse_sketch("Use a hashmap to count occurrences.")
    assert ok
    assert "hashmap" in s


# ============================================================
# parse_code
# ============================================================

def test_code_basic_fenced():
    c, ok = parse_code("def f(x):\n    return x + 1\n```\n### ")
    assert ok
    assert "def f" in c
    assert "```" not in c


def test_code_with_leading_python_marker():
    c, ok = parse_code("```python\ndef g():\n    return 0\n```")
    assert ok
    assert c.startswith("def g")


def test_code_unterminated_runs_to_header():
    c, ok = parse_code("import sys\ndata = sys.stdin.read()\nprint(data)\n### ")
    assert ok
    assert "###" not in c


def test_code_empty():
    c, ok = parse_code("```")
    assert not ok


def test_code_no_code_keywords_marks_fail():
    c, ok = parse_code("just some prose here\n```")
    assert not ok
