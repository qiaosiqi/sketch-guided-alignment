"""
Pilot 采样:50 题 × 100 sketch × 3 温度 (0.4, 0.7, 1.0),每个有效 sketch 配 1 code,
然后执行,得 pass_ratio。

输出:
    {out_dir}/sketches.jsonl
    {out_dir}/codes.jsonl
    {out_dir}/exec.jsonl

跑完后请用 03_analyze_pilot.py 分析。

用法:
    python -m v2.scripts.02_sample_pilot \
        --problems_jsonl out/apps/train.jsonl \
        --model_path /path/to/StarCoder2-3B \
        --out_dir out/pilot \
        --n_problems 50 \
        --n_per_temp 100 \
        --temps 0.4 0.7 1.0
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
from dataclasses import asdict

from ..data.schema import Problem, ExecutionResult
from ..sampling import build_backend, sample_sketches, sample_codes
from ..execution import run_one_solution


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("pilot")


def load_problems(path: str) -> list[Problem]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(Problem(**json.loads(line)))
    return out


def select_pilot_problems(all_probs: list[Problem], n: int, seed: int) -> list[Problem]:
    """从面试级问题中随机抽 n 道作 pilot。

    数据集已只含 interview(competition 通过率近零被排除,见 apps_loader),
    故不再做难度分层,直接固定 seed 随机抽样。
    """
    pool = list(all_probs)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def run_executions(codes_path: str, problems_by_id: dict[str, Problem], out_path: str):
    """逐条读 codes.jsonl,对 parsed_ok 的 code 跑 execution,append 写出 exec.jsonl。"""
    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_keys.add((obj["task_id"], obj["sample_id"], obj["code_id"]))
                except Exception:
                    pass

    with open(codes_path, "r", encoding="utf-8") as f, open(out_path, "a", encoding="utf-8") as fout:
        for i, line in enumerate(f, 1):
            c = json.loads(line)
            key = (c["task_id"], c["sample_id"], c.get("code_id", 0))
            if key in done_keys:
                continue
            if not c.get("parsed_ok"):
                # 不能跑的 code 也写一条 0 分记录
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
                    do_timing=False,        # pilot 不测时,只看 pass_ratio
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
            if i % 50 == 0:
                log.info(f"executed {i} codes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems_jsonl", required=True, help="train.jsonl from apps_loader")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_problems", type=int, default=50)
    ap.add_argument("--n_per_temp", type=int, default=100)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.4, 0.7, 1.0])
    ap.add_argument("--prefer_backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--max_new_tokens_sketch", type=int, default=256)
    ap.add_argument("--max_new_tokens_code", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1)
    # 数据并行分片:在 5090×2 上,两个进程各占一卡,各跑题目子集,产物事后 cat 合并
    ap.add_argument("--shard_id", type=int, default=0, help="本进程负责的分片编号 [0, n_shards)")
    ap.add_argument("--n_shards", type=int, default=1, help="总分片数;1=不分片")
    args = ap.parse_args()
    assert 0 <= args.shard_id < args.n_shards, "shard_id 必须在 [0, n_shards) 内"

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 选题
    all_probs = load_problems(args.problems_jsonl)
    chosen = select_pilot_problems(all_probs, args.n_problems, args.seed)
    # 数据并行:在固定 seed shuffle 后的列表上做 stride 切片,两 shard 之间题目不重叠
    if args.n_shards > 1:
        chosen = chosen[args.shard_id::args.n_shards]
        log.info(f"shard {args.shard_id}/{args.n_shards}: {len(chosen)} problems after slicing")
    problems_by_id = {p.task_id: p for p in chosen}
    with open(os.path.join(args.out_dir, "chosen_problems.jsonl"), "w", encoding="utf-8") as f:
        for p in chosen:
            f.write(json.dumps({"task_id": p.task_id, "difficulty": p.difficulty,
                                "io_format": p.io_format, "n_tests": len(p.inputs)},
                               ensure_ascii=False) + "\n")
    log.info(f"selected {len(chosen)} problems for pilot")

    # 2) Sketch 采样
    # bfloat16:与 StarCoder2 原生权重 + bf16 训练一致;Blackwell 原生支持
    backend = build_backend(args.model_path, dtype="bfloat16", prefer=args.prefer_backend)

    # 拿个 tokenizer 做长度过滤(直接用 backend 内部的)
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        # vllm 后端没暴露 tokenizer,自己加载一个
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sketches_path = os.path.join(args.out_dir, "sketches.jsonl")
    sample_sketches(
        backend=backend, problems=chosen,
        n_per_temp=args.n_per_temp, temps=args.temps,
        out_path=sketches_path, tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens_sketch,
        seed=args.seed,
    )
    log.info("sketch sampling done")

    # 3) Code 采样(温度沿用 sketch 的温度信息,但 Stage-2 我们统一用 0.6,避免再爆炸)
    codes_path = os.path.join(args.out_dir, "codes.jsonl")
    sample_codes(
        backend=backend,
        problems_by_id=problems_by_id,
        sketches_path=sketches_path,
        out_path=codes_path,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens_code,
        temperature=0.6,
        seed=args.seed,
    )
    log.info("code sampling done")

    backend.shutdown()

    # 4) Execution
    exec_path = os.path.join(args.out_dir, "exec.jsonl")
    run_executions(codes_path, problems_by_id, exec_path)
    log.info("execution done")


if __name__ == "__main__":
    main()
