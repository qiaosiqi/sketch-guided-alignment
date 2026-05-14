"""
所有 prompt 模板。两段式采样的核心契约。

设计要点:
- Sketch prompt:让模型只输出 2-4 句英文 algorithmic sketch,严格用标记包裹便于解析
- Code prompt:把 sketch 作为已知条件,要求模型只输出代码,用 ```python ... ``` 包裹
- 不混用一段式 prompt(因为 v2 走两段式生成路线)
- few-shot 示例需要覆盖 stdio 和 fncall 两种 IO 格式,否则模型在 APPS stdio 题上易跑偏
"""
from textwrap import dedent
from .schema import Problem


# ============================================================
# Sketch prompt
# ============================================================

SKETCH_SYSTEM = dedent("""\
    You are an expert competitive programmer. For each programming problem,
    write ONLY a short algorithmic sketch (2 to 4 sentences) describing the
    high-level approach. Do not write code. Do not restate the problem.
    Wrap your sketch strictly between <SKETCH> and </SKETCH> tags.
""").strip()


SKETCH_FEWSHOT = dedent("""\
    ### Problem
    Given an integer n, return the n-th Fibonacci number (F(0)=0, F(1)=1).
    Input is a single integer on stdin; print the answer.

    ### Sketch
    <SKETCH>
    Use bottom-up dynamic programming. Maintain two variables a=F(0), b=F(1)
    and iterate n times, updating (a, b) = (b, a+b). Print a at the end.
    Time O(n), space O(1). Handle n=0 by returning a directly.
    </SKETCH>

    ### Problem
    Implement a function `is_palindrome(s: str) -> bool` that returns whether
    the string reads the same forwards and backwards.

    ### Sketch
    <SKETCH>
    Compare the string with its reverse using slicing s == s[::-1].
    Constant extra memory beyond the reversed string; O(n) time. No edge
    cases needed since empty strings naturally compare equal.
    </SKETCH>
""").strip()


def build_sketch_prompt(problem: Problem) -> str:
    """组装 sketch 阶段的完整 prompt。返回纯文本(不带 chat template)。"""
    io_hint = (
        f"You will write a function named `{problem.fn_name}` that takes the listed arguments."
        if problem.io_format == "fncall"
        else "Input is given on stdin; output is printed to stdout."
    )
    return dedent(f"""\
        {SKETCH_SYSTEM}

        {SKETCH_FEWSHOT}

        ### Problem
        {problem.question.strip()}

        Note: {io_hint}

        ### Sketch
        <SKETCH>
        """)


# ============================================================
# Code prompt
# ============================================================

CODE_SYSTEM = dedent("""\
    You are an expert competitive programmer. You are given a problem and
    a high-level algorithmic sketch. Your task is to implement the sketch
    in Python. Output ONLY the Python code wrapped strictly in a fenced
    code block. Do not include explanations.
""").strip()


CODE_FEWSHOT = dedent("""\
    ### Problem
    Given an integer n, return the n-th Fibonacci number (F(0)=0, F(1)=1).
    Input is a single integer on stdin; print the answer.

    ### Sketch
    Use bottom-up dynamic programming. Maintain two variables a=F(0), b=F(1)
    and iterate n times, updating (a, b) = (b, a+b). Print a at the end.

    ### Code
    ```python
    n = int(input())
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    print(a)
    ```

    ### Problem
    Implement a function `is_palindrome(s: str) -> bool` that returns whether
    the string reads the same forwards and backwards.

    ### Sketch
    Compare the string with its reverse using slicing s == s[::-1].

    ### Code
    ```python
    def is_palindrome(s: str) -> bool:
        return s == s[::-1]
    ```
""").strip()


def build_code_prompt(problem: Problem, sketch: str) -> str:
    """组装 code 阶段的完整 prompt。问题 + sketch 作为条件,要求输出代码。"""
    io_hint = (
        f"Implement a function named `{problem.fn_name}`."
        if problem.io_format == "fncall"
        else "Read from stdin and print to stdout."
    )
    starter = ""
    if problem.starter_code:
        starter = f"\n    Starter code:\n    ```python\n    {problem.starter_code.strip()}\n    ```\n"

    return dedent(f"""\
        {CODE_SYSTEM}

        {CODE_FEWSHOT}

        ### Problem
        {problem.question.strip()}

        Note: {io_hint}{starter}

        ### Sketch
        {sketch.strip()}

        ### Code
        ```python
        """)


# ============================================================
# Training prompt (拼成一段式格式,供 SFT/DPO 训练用)
# ============================================================

TRAINING_TEMPLATE = dedent("""\
    ### Problem
    {question}

    ### Sketch
    {sketch}

    ### Code
    ```python
    {code}
    ```""")


def build_training_text(question: str, sketch: str, code: str) -> str:
    """SFT/DPO 训练样本的标准格式。response 起始用于 collator 切 label。"""
    return TRAINING_TEMPLATE.format(
        question=question.strip(),
        sketch=sketch.strip(),
        code=code.strip(),
    )


# ============================================================
# Response template (DataCollatorForCompletionOnlyLM 切 label 用)
# ============================================================

# loss 只在 "### Sketch" 之后开始算。即模型需要学会从 problem 推出 sketch 和 code。
RESPONSE_TEMPLATE = "\n### Sketch\n"
