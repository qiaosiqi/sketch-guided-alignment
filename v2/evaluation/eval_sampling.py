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
# 必须在 transformers / tokenizers / sampling import 之前设;否则 execution 阶段
# fork 子进程时会触发 tokenizers 警告并自禁用并行。
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging

from ..data.schema import Problem
from ..sampling import build_backend, sample_sketches, sample_codes
from ..execution import run_executions_parallel


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("eval_sample")


def load_problems(path: str) -> list[Problem]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(Problem(**json.loads(line)))
    return out


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
    # 数据并行分片(对应 5090×2 拓扑)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument(
        "--exec_workers", type=int, default=None,
        help="execution 阶段线程并发数,默认按 (cpu_count-4)/n_shards 推算",
    )
    args = ap.parse_args()
    assert 0 <= args.shard_id < args.n_shards, "shard_id 必须在 [0, n_shards) 内"

    os.makedirs(args.out_dir, exist_ok=True)

    problems = load_problems(args.problems_jsonl)
    if args.n_shards > 1:
        # 按 task_id 排序后 stride 切片,确保两 shard 在不同 model 复跑时分片一致
        problems = sorted(problems, key=lambda p: p.task_id)[args.shard_id::args.n_shards]
        log.info(f"shard {args.shard_id}/{args.n_shards}: {len(problems)} problems after slicing")
    problems_by_id = {p.task_id: p for p in problems}
    log.info(f"loaded {len(problems)} test problems")

    # bfloat16 与 StarCoder2 原生权重、训练 dtype 一致;Blackwell 同样原生支持
    backend = build_backend(args.model_path, dtype="bfloat16", prefer=args.prefer_backend)
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
    os.environ.setdefault("V2_N_SHARDS", str(args.n_shards))
    run_executions_parallel(
        codes_path=codes_path,
        problems_by_id=problems_by_id,
        out_path=exec_path,
        do_timing=args.do_timing,
        timeout_per_test=3.0,
        max_workers=args.exec_workers,
    )
    log.info("eval sampling done")


if __name__ == "__main__":
    main()
