"""data/prompts.py 基本组装的 smoke tests。"""
from v2.data.prompts import (
    build_sketch_prompt, build_code_prompt, build_training_text,
    RESPONSE_TEMPLATE,
)


def test_sketch_prompt_stdio(stdio_problem):
    p = build_sketch_prompt(stdio_problem)
    assert "<SKETCH>" in p
    assert "stdin" in p
    assert "Read N from stdin" in p


def test_sketch_prompt_fncall(fncall_problem):
    p = build_sketch_prompt(fncall_problem)
    assert "add" in p
    assert "function" in p.lower()


def test_code_prompt_includes_sketch(fncall_problem):
    sketch = "Just return a + b directly."
    p = build_code_prompt(fncall_problem, sketch)
    assert sketch in p
    assert "```python" in p


def test_code_prompt_with_starter(fncall_problem):
    fncall_problem.starter_code = "def add(a, b):\n    "
    p = build_code_prompt(fncall_problem, "trivial sum")
    assert "Starter code" in p


def test_training_text_has_response_template():
    t = build_training_text("Q", "S", "C")
    assert RESPONSE_TEMPLATE in t
    assert "### Code" in t
    assert "```python" in t


def test_response_template_at_correct_position():
    """模型应该从 ### Sketch 开始预测,所以 RESPONSE_TEMPLATE 必须出现在 question 之后。"""
    t = build_training_text("the_question", "the_sketch", "the_code")
    q_pos = t.find("the_question")
    rt_pos = t.find(RESPONSE_TEMPLATE)
    assert q_pos < rt_pos
    sk_pos = t.find("the_sketch")
    assert rt_pos < sk_pos
