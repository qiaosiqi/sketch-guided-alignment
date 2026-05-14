"""
评测阶段:在 test 集上生成 → 执行 → 计算指标。

跟 pilot 不一样的地方:
- 不评分(不调 GLM-4-Air),除非显式 --score
- test 集上 sample 100 个解 / 题,sketch+code 一次性走完
- 测时:全 pass 的解才测,沿用 timing.py 的 CoV ≤ 0.1

输出:
    {out_dir}/sketches.jsonl
    {out_dir}/codes.jsonl
    {out_dir}/exec.jsonl  # 含 runtime_ns_mean
    {out_dir}/scores.jsonl (可选)
"""
from __future__ import annotations
import argparse
import json
import logging
import os
from dataclasses import asdict

from ..data.schema import Problem, ExecutionResult
from ..sampling import build_backend, sample_sketches, sample_codes
from ..execution import run_one_solution


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("eval_sample")


def load_problems(path: str) -> list[Problem]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(Problem(**json.loads(line)))
    return out


def run_executions(codes_path: str, problems_by_id: dict[str, Problem],
                   out_path: str, do_timing: bool = True):
    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_keys.add((obj["task_id"], obj["sample_id"], obj.get("code_id", 0)))
                except Exception:
                    pass

    with open(codes_path, "r", encoding="utf-8") as f, open(out_path, "a", encoding="utf-8") as fout:
        for i, line in enumerate(f, 1):
            c = json.loads(line)
            key = (c["task_id"], c["sample_id"], c.get("code_id", 0))
            if key in done_keys:
                continue
            if not c.get("parsed_ok"):
                er = ExecutionResult(
                    task_id=c["task_id"], sample_id=c["sample_id"], code_id=c.get("code_id", 0),
                    n_tests=0, n_passed=0, pass_ratio=0.0,
                    per_test_pass=[], error="unparsable_code",
                )
                fout.write(json.dumps(asdict(er), ensure_ascii=False) + "\n")
                continue
            problem = problems_by_id.get(c["task_id"])
            if problem is None:
                continue
            try:
                er = run_one_solution(
                    problem, c["code"],
                    timeout_per_test=3.0,
                    do_timing=do_timing,
                    sample_id=c["sample_id"],
                    code_id=c.get("code_id", 0),
                )
            except Exception as e:
                er = ExecutionResult(
                    task_id=c["task_id"], sample_id=c["sample_id"], code_id=c.get("code_id", 0),
                    n_tests=len(problem.inputs), n_passed=0, pass_ratio=0.0,
                    per_test_pass=[False] * len(problem.inputs), error=f"runner_crash: {e}",
                )
            fout.write(json.dumps(asdict(er), ensure_ascii=False) + "\n")
            fout.flush()
            if i % 100 == 0:
                log.info(f"executed {i} codes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems_jsonl", required=True, help="test.jsonl from apps_loader")
    ap.add_argument("--model_path", required=True, help="可以是 base model 或 SFT/DPO 后的 checkpoint")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_temp", type=int, default=100)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.6])
    ap.add_argument("--prefer_backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--max_new_tokens_sketch", type=int, default=256)
    ap.add_argument("--max_new_tokens_code", type=int, default=1024)
    ap.add_argument("--do_timing", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    problems = load_problems(args.problems_jsonl)
    problems_by_id = {p.task_id: p for p in problems}
    log.info(f"loaded {len(problems)} test problems")

    backend = build_backend(args.model_path, dtype="float16", prefer=args.prefer_backend)
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sketches_path = os.path.join(args.out_dir, "sketches.jsonl")
    sample_sketches(
        backend=backend, problems=problems,
        n_per_temp=args.n_per_temp, temps=args.temps,
        out_path=sketches_path, tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens_sketch,
        seed=args.seed,
    )
    codes_path = os.path.join(args.out_dir, "codes.jsonl")
    sample_codes(
        backend=backend, problems_by_id=problems_by_id,
        sketches_path=sketches_path, out_path=codes_path,
        tokenizer=tokenizer, max_new_tokens=args.max_new_tokens_code,
        temperature=0.6, seed=args.seed,
    )
    backend.shutdown()

    exec_path = os.path.join(args.out_dir, "exec.jsonl")
    run_executions(codes_path, problems_by_id, exec_path, do_timing=args.do_timing)
    log.info("eval sampling done")


if __name__ == "__main__":
    main()
