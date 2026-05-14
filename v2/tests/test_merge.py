"""merge/build_dataset.py 的 smoke tests。

构造 fake sketches/codes/exec/scores jsonl,验证合并产出。
"""
import json
from pathlib import Path

from v2.data.schema import Problem
from v2.merge.build_dataset import build_merged


def _write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_build_merged_basic(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()

    # 一题两个候选,其中一个 parsed_ok=False
    codes = [
        {"task_id": "p1", "sample_id": 0, "code_id": 0, "sketch": "s0", "code": "c0",
         "sample_temp": 0.6, "parsed_ok": True},
        {"task_id": "p1", "sample_id": 1, "code_id": 0, "sketch": "s1", "code": "c1",
         "sample_temp": 0.6, "parsed_ok": True},
        {"task_id": "p1", "sample_id": 2, "code_id": 0, "sketch": "", "code": "",
         "sample_temp": 0.6, "parsed_ok": False},
    ]
    execs = [
        {"task_id": "p1", "sample_id": 0, "code_id": 0,
         "n_tests": 3, "n_passed": 3, "pass_ratio": 1.0,
         "per_test_pass": [True]*3, "runtime_ns_mean": 100.0, "runtime_ns_std": 10.0},
        {"task_id": "p1", "sample_id": 1, "code_id": 0,
         "n_tests": 3, "n_passed": 1, "pass_ratio": 1/3,
         "per_test_pass": [True, False, False]},
    ]
    scores = [
        {"task_id": "p1", "sample_id": 0, "code_id": 0, "placeholder": None,
         "scores": {
             "sketch": {"S1_correctness": 8, "S2_specificity": 7,
                        "S3_complexity_awareness": 6, "S4_edge_coverage": 7,
                        "raw_response": ""},
             "code": {"C1_faithfulness": 9, "C2_time_complexity": 8,
                      "C3_space_complexity": 7, "C4_readability": 6,
                      "C5_edge_handling": 7, "raw_response": ""},
             "final": 7.4, "judge_model": "GLM-4-Air", "parsable": True,
         }},
    ]
    _write_jsonl(sample_dir / "codes.jsonl", codes)
    _write_jsonl(sample_dir / "exec.jsonl", execs)
    _write_jsonl(sample_dir / "scores.jsonl", scores)

    problem = Problem(
        task_id="p1", difficulty="interview", io_format="fncall",
        question="Q1", inputs=[[1]], outputs=[[1]], fn_name="f",
    )
    out_path = tmp_path / "merged.jsonl"
    build_merged({"p1": problem}, str(sample_dir), str(out_path))

    assert out_path.exists()
    rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    assert len(rows) == 1
    m = rows[0]
    assert m["task_id"] == "p1"
    # 只保留 parsed_ok=True 的 2 个
    assert len(m["answers"]) == 2
    a0 = m["answers"][0]
    assert a0["pass_ratio"] == 1.0
    assert a0["runtime_ns_mean"] == 100.0
    assert a0["algo_final"] == 7.4
    # 第二个解没评分,algo_final=-1
    assert m["answers"][1]["algo_final"] == -1.0


def test_build_merged_drops_problem_with_no_valid_answers(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()

    codes = [
        {"task_id": "p1", "sample_id": 0, "code_id": 0,
         "sketch": "", "code": "", "sample_temp": 0.6, "parsed_ok": False},
    ]
    _write_jsonl(sample_dir / "codes.jsonl", codes)
    _write_jsonl(sample_dir / "exec.jsonl", [])
    _write_jsonl(sample_dir / "scores.jsonl", [])

    problem = Problem(
        task_id="p1", difficulty="interview", io_format="fncall",
        question="Q1", inputs=[[1]], outputs=[[1]], fn_name="f",
    )
    out_path = tmp_path / "merged.jsonl"
    build_merged({"p1": problem}, str(sample_dir), str(out_path))

    rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    assert rows == []
