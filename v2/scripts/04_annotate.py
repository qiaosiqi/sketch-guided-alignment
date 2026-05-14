"""
对采样产生的 (sketch, code) 做 GLM-4-Air 评分。

前置:
    out/{run_name}/codes.jsonl  (来自 sample_codes)
    out/{run_name}/exec.jsonl   (来自 execution)
环境变量:
    GLM_API_KEY=...

用法:
    python -m v2.scripts.04_annotate \
        --problems_jsonl out/apps/train.jsonl \
        --sample_dir out/main \
        --pass_threshold 0.0 \
        --alpha 0.4
"""
from __future__ import annotations
import argparse
import logging
import os

from ..annotation import build_judge, batch_annotate
from ..merge.build_dataset import load_problems


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems_jsonl", required=True)
    ap.add_argument("--sample_dir", required=True, help="目录含 codes.jsonl, exec.jsonl;输出 scores.jsonl")
    ap.add_argument("--pass_threshold", type=float, default=0.0,
                    help="只对 pass_ratio >= 此值的 code 评分。0 = 全部评。0.8 = 只评接近全 pass")
    ap.add_argument("--alpha", type=float, default=0.4, help="sketch 权重(final = α·S + (1-α)·C)")
    ap.add_argument("--model", default="glm-4-air")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--sleep_between", type=float, default=0.0, help="速率限制可用")
    args = ap.parse_args()

    problems = load_problems(args.problems_jsonl)
    judge = build_judge(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)

    codes_path = os.path.join(args.sample_dir, "codes.jsonl")
    exec_path = os.path.join(args.sample_dir, "exec.jsonl")
    out_path = os.path.join(args.sample_dir, "scores.jsonl")

    batch_annotate(
        judge=judge,
        codes_path=codes_path,
        exec_path=exec_path,
        problems_by_id=problems,
        out_path=out_path,
        pass_threshold=args.pass_threshold,
        alpha=args.alpha,
        sleep_between=args.sleep_between,
    )


if __name__ == "__main__":
    main()
