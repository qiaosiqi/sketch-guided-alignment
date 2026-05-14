"""
9 维评分 rubric。两次独立调用 judge:
    score_sketch  — 评 sketch 4 维 (S1..S4)
    score_code    — 评 code 5 维 (C1..C5),给定 sketch 作为参考

设计要点:
- Sketch 评分**不给代码**,避免代码影响 sketch 打分(消除偏差)
- Code 评分**给 sketch**,因为 C1 (sketch faithfulness) 必须对比
- Output 严格要求 JSON,容错解析 / 失败标记
- Prompt 内嵌每维度的 0-10 锚定(rubric),让 judge 输出更稳定
- 代码题面超过 2000 字符时截断(供 judge 用,不影响训练)
"""
from __future__ import annotations
import json
import re
from typing import Optional

from .judge_client import GLMJudge, JudgeError
from ..data.schema import SketchScores, CodeScores


PROBLEM_TRUNC = 2000
CODE_TRUNC = 2000


# ============================================================
# Sketch rubric
# ============================================================

SKETCH_SYSTEM = (
    "You are a senior algorithms judge. You evaluate algorithmic SKETCHES "
    "(short natural-language descriptions of how to solve a programming problem). "
    "You give an integer score from 0 to 10 on each of four dimensions. "
    "You output your scores as a single JSON object on the LAST line."
)

SKETCH_RUBRIC = """\
Rate the sketch on these 4 dimensions, integer 0-10 each:

S1 (Correctness): does the described algorithm correctly solve the problem?
  0 = wrong approach, will never work
  5 = right intuition but a key step is missing or incorrect
  10 = clearly correct, complete algorithm

S2 (Specificity): is the sketch concrete enough to translate into code?
  0 = vague hand-waving ("use a loop")
  5 = names approach without details (says "DP" but no states)
  10 = names data structures, key variables, transition rules

S3 (Complexity awareness): does it acknowledge time/space cost?
  0 = silent on complexity
  5 = mentions one dimension
  10 = states both time and space with brief justification

S4 (Edge coverage): does it mention boundary / corner cases?
  0 = ignores edges entirely
  5 = mentions one obvious edge
  10 = covers boundary, empty, single-element, max-size cases as relevant

First write a 1-2 sentence analysis (no headers).
Then on the FINAL line output ONLY this JSON (no trailing text):
{"S1": <int>, "S2": <int>, "S3": <int>, "S4": <int>}
"""


def _build_sketch_user(problem_text: str, sketch: str) -> str:
    p = problem_text[:PROBLEM_TRUNC]
    return (
        f"Problem:\n{p}\n\n"
        f"Sketch:\n{sketch.strip()}\n\n"
        f"{SKETCH_RUBRIC}"
    )


# ============================================================
# Code rubric
# ============================================================

CODE_SYSTEM = (
    "You are a senior code reviewer for competitive programming. You evaluate Python "
    "CODE submissions against the problem statement and a provided algorithmic sketch. "
    "You give an integer score from 0 to 10 on each of five dimensions. "
    "You output your scores as a single JSON object on the LAST line."
)

CODE_RUBRIC = """\
Rate the code on these 5 dimensions, integer 0-10 each:

C1 (Sketch faithfulness): does the code implement the described sketch?
  0 = code does something completely different (sketch says DP but code is brute force)
  5 = partially follows sketch but with major deviations
  10 = faithful implementation, matches sketch closely

C2 (Time complexity optimality):
  0 = will obviously TLE on the constraints (e.g. O(n^3) when n=1e5)
  5 = passable but suboptimal
  10 = optimal Big-O for this problem

C3 (Space complexity optimality):
  0 = wasteful (allocates O(n^2) when O(n) suffices)
  5 = passable
  10 = optimal

C4 (Readability):
  0 = unreadable, no naming, dense one-liners
  5 = readable with effort
  10 = clear names, structure, idiomatic Python

C5 (Edge handling):
  0 = no edge case handling, will crash on boundary
  5 = handles obvious edges
  10 = handles all relevant edges (empty input, max size, single element, ...)

First write a 1-2 sentence analysis (no headers).
Then on the FINAL line output ONLY this JSON (no trailing text):
{"C1": <int>, "C2": <int>, "C3": <int>, "C4": <int>, "C5": <int>}
"""


def _build_code_user(problem_text: str, sketch: str, code: str) -> str:
    p = problem_text[:PROBLEM_TRUNC]
    c = code[:CODE_TRUNC]
    return (
        f"Problem:\n{p}\n\n"
        f"Sketch:\n{sketch.strip()}\n\n"
        f"Code:\n```python\n{c}\n```\n\n"
        f"{CODE_RUBRIC}"
    )


# ============================================================
# 解析
# ============================================================

# 匹配最后一行(或最后一个) JSON 对象
_JSON_LAST = re.compile(r"\{[^{}]*\}\s*$", re.MULTILINE | re.DOTALL)


def _extract_json_scores(text: str, required_keys: tuple[str, ...]) -> Optional[dict]:
    """从 judge 输出里抽 JSON 分数对象。失败返回 None。"""
    candidates: list[str] = []
    # 优先匹配末尾
    m = _JSON_LAST.search(text)
    if m:
        candidates.append(m.group(0))
    # 再扫所有 { ... },选最后一个能成功 parse 出全键的
    candidates += re.findall(r"\{[^{}]*\}", text)
    for c in reversed(candidates):
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if all(k in obj for k in required_keys):
            try:
                cleaned = {k: float(obj[k]) for k in required_keys}
            except (TypeError, ValueError):
                continue
            # clamp 到 [0, 10]
            for k in cleaned:
                cleaned[k] = max(0.0, min(10.0, cleaned[k]))
            return cleaned
    return None


# ============================================================
# 公开接口
# ============================================================

def score_sketch(judge: GLMJudge, problem_text: str, sketch: str) -> SketchScores | None:
    """评 sketch。失败返回 None。"""
    user = _build_sketch_user(problem_text, sketch)
    try:
        resp = judge.chat(SKETCH_SYSTEM, user)
    except JudgeError:
        return None
    scores = _extract_json_scores(resp, ("S1", "S2", "S3", "S4"))
    if scores is None:
        return None
    return SketchScores(
        S1_correctness=scores["S1"],
        S2_specificity=scores["S2"],
        S3_complexity_awareness=scores["S3"],
        S4_edge_coverage=scores["S4"],
        raw_response=resp,
    )


def score_code(judge: GLMJudge, problem_text: str, sketch: str, code: str) -> CodeScores | None:
    """评 code。失败返回 None。"""
    user = _build_code_user(problem_text, sketch, code)
    try:
        resp = judge.chat(CODE_SYSTEM, user)
    except JudgeError:
        return None
    scores = _extract_json_scores(resp, ("C1", "C2", "C3", "C4", "C5"))
    if scores is None:
        return None
    return CodeScores(
        C1_faithfulness=scores["C1"],
        C2_time_complexity=scores["C2"],
        C3_space_complexity=scores["C3"],
        C4_readability=scores["C4"],
        C5_edge_handling=scores["C5"],
        raw_response=resp,
    )
