"""
评测入口。两步:eval_sampling 采样+执行,然后 metrics 算指标。

用法:
    # 一步式:
    python -m v2.scripts.08_eval \
        --problems_jsonl out/apps/test.jsonl \
        --model_path out/runs/dpo_gvb_from_sft \
        --out_dir out/evals/dpo_gvb \
        --do_timing

    # 也可以分步:先采样
    python -m v2.evaluation.eval_sampling --problems_jsonl ... --model_path ... --out_dir ...
    # 再算指标
    python -m v2.evaluation.metrics --exec_path out/evals/dpo_gvb/exec.jsonl
"""
import argparse
import os
import subprocess
import sys

from v2.evaluation.metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems_jsonl", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_temp", type=int, default=100)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.6])
    ap.add_argument("--prefer_backend", default="vllm")
    ap.add_argument("--do_timing", action="store_true")
    ap.add_argument("--scores_path", default=None)
    args = ap.parse_args()

    # Step 1: sampling + execution
    cmd = [
        sys.executable, "-m", "v2.evaluation.eval_sampling",
        "--problems_jsonl", args.problems_jsonl,
        "--model_path", args.model_path,
        "--out_dir", args.out_dir,
        "--n_per_temp", str(args.n_per_temp),
        "--temps", *[str(t) for t in args.temps],
        "--prefer_backend", args.prefer_backend,
    ]
    if args.do_timing:
        cmd.append("--do_timing")
    print(" ".join(cmd))
    subprocess.check_call(cmd)

    # Step 2: metrics
    exec_path = os.path.join(args.out_dir, "exec.jsonl")
    metrics = compute_metrics(exec_path, args.scores_path)
    import json
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
