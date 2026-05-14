"""
Pytest fixtures for v2 smoke tests.

无 GPU / 无 API / 无大模型权重依赖。
所有 fixture 都构造在内存里,或者用 tmp_path 写到临时文件夹。
"""
from __future__ import annotations
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from v2.data.schema import Problem


IS_WINDOWS = sys.platform == "win32"


# ============================================================
# Fixtures: 题目
# ============================================================

@pytest.fixture
def fncall_problem() -> Problem:
    """函数调用式题目:add(a, b)。"""
    return Problem(
        task_id="test_add",
        difficulty="interview",
        io_format="fncall",
        question="Write a function add(a, b) that returns a + b.",
        inputs=[[1, 2], [3, 4], [10, -5]],
        outputs=[[3], [7], [5]],
        fn_name="add",
    )


@pytest.fixture
def stdio_problem() -> Problem:
    """stdin/stdout 式题目:把第一行的 N 加 1 输出。"""
    return Problem(
        task_id="test_plus_one",
        difficulty="interview",
        io_format="stdio",
        question="Read N from stdin, print N+1.",
        inputs=[["5"], ["100"], ["-3"]],
        outputs=[["6"], ["101"], ["-2"]],
        fn_name=None,
    )


# ============================================================
# Fixtures: 临时 APPS-style 目录
# ============================================================

@pytest.fixture
def fake_apps_root(tmp_path: Path) -> Path:
    """
    构造一个最小 APPS 目录结构,3 题:
        train/0001  competition  fncall
        train/0002  interview    stdio
        train/0003  introductory(应被过滤)
    """
    root = tmp_path / "apps_raw"
    train = root / "train"
    train.mkdir(parents=True)

    def _make(idx: str, difficulty: str, io: dict, question: str, starter: str | None = None):
        d = train / idx
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"difficulty": difficulty}), encoding="utf-8")
        (d / "input_output.json").write_text(json.dumps(io), encoding="utf-8")
        (d / "question.txt").write_text(question, encoding="utf-8")
        (d / "solutions.json").write_text(json.dumps([]), encoding="utf-8")
        if starter:
            (d / "starter_code.py").write_text(starter, encoding="utf-8")

    _make("0001", "competition",
          {"fn_name": "twice", "inputs": [[3]], "outputs": [[6]]},
          "Twice the input.", "def twice(x):\n    ")
    _make("0002", "interview",
          {"inputs": [["7"]], "outputs": [["8"]]},
          "Print n+1.")
    _make("0003", "introductory",
          {"inputs": [["1"]], "outputs": [["1"]]},
          "Trivial.")

    # 也建一个空 test 目录避免 iter_split 抛 FileNotFoundError
    (root / "test").mkdir()
    return root


# ============================================================
# Fixtures: merged.jsonl 内存样本
# ============================================================

def _ans(sketch: str, code: str, pass_ratio: float, runtime: float | None,
         algo_final: float, sample_temp: float = 0.6) -> dict:
    return {
        "sketch": sketch, "code": code,
        "pass_ratio": pass_ratio,
        "n_tests": 3,
        "runtime_ns_mean": runtime,
        "runtime_ns_std": None,
        "algo_final": algo_final,
        "algo_breakdown": None,
        "sample_temp": sample_temp,
    }


@pytest.fixture
def merged_two_problems(tmp_path: Path) -> Path:
    """两道题,每题 6 个候选,覆盖 hvl/qvs/gvb 三类 pair 都能构造。"""
    p1 = {
        "task_id": "p1", "difficulty": "interview", "io_format": "fncall",
        "question": "Q1", "fn_name": "f",
        "answers": [
            _ans("s_good1", "c_good1", 1.0, 1000.0, 9.0),
            _ans("s_good2", "c_good2", 1.0, 2000.0, 8.5),
            _ans("s_ok",    "c_ok",    0.95, 3000.0, 5.0),
            _ans("s_bad1",  "c_bad1",  0.85, None, 3.0),
            _ans("s_lo",    "c_lo",    0.2, None, -1.0),
            _ans("s_zero",  "c_zero",  0.0, None, -1.0),
        ],
    }
    p2 = {
        "task_id": "p2", "difficulty": "competition", "io_format": "stdio",
        "question": "Q2", "fn_name": None,
        "answers": [
            _ans("a_g", "b_g", 1.0, 500.0, 7.5),
            _ans("a_b", "b_b", 0.9, None, 2.0),
            _ans("a_l", "b_l", 0.1, None, -1.0),
        ],
    }
    f = tmp_path / "merged.jsonl"
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(p1) + "\n")
        fh.write(json.dumps(p2) + "\n")
    return f


# ============================================================
# Fixtures: 假 tokenizer(不要拉网络/不依赖模型)
# ============================================================

class FakeTokenizer:
    """最小 tokenizer。按字符级别(模仿 BPE 失败时仍能跑测试)。"""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) % 256 + 10 for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def __call__(self, text, truncation=False, padding=False, max_length=None,
                 add_special_tokens=True, return_tensors=None, **kwargs):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()
