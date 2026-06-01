"""
为"测试集 rubric 打分"做子采样,避免对全部解调 GLM-4-Air(成本炸弹)。

背景:
    08_eval 在 test 集上每题采样很多解(base/sft ~100、dpo ~50)。若对全部解打分,
    每条解 2 次 API 调用,6 个模型合计数百万次调用,既慢又贵且没必要。
    本脚本固定**同一批随机 task_id**(跨模型配对比较),每题随机封顶 M 条解,
    把过滤后的 codes/exec 写到 {evals_dir}/{model}/{out_subdir}/,
    随后只对该子目录跑 04_annotate + metrics。

不引入偏差:
    - 选题:取所有模型 codes.jsonl 中 task_id 的交集,固定 seed 抽 N 题(配对)。
    - 选解:每题对该模型的解固定 seed 随机抽 ≤M 条 → mean_algo_final 的无偏估计。
    - 选题集对所有模型一致;每题的解抽样 seed 由 (seed, model, task_id) 派生,
      跨平台/重跑可复现(用 hashlib,不用 Python 内置 hash 的随机化盐)。

用法:
    python -m v2.scripts.subsample_for_scoring \
        --evals_dir /root/shared-nvme/work/out/evals \
        --models base sft_alg_top25 dpo_pvf dpo_qvs dpo_gvb dpo_all \
        --n_problems 400 --max_per_problem 10 --seed 1
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("subsample")


def _key(o: dict) -> tuple:
    return (o["task_id"], o["sample_id"], o.get("code_id", 0))


def _derive_seed(*parts) -> int:
    """由 (seed, model, task_id) 派生稳定整数种子(不依赖 PYTHONHASHSEED)。"""
    h = hashlib.sha256("\x00".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: str, records: list[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals_dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n_problems", type=int, default=400)
    ap.add_argument("--max_per_problem", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_subdir", default="sub")
    args = ap.parse_args()

    # 1) 读每个模型的 codes,按 task_id 分组,记录各模型出现的 task_id 集合
    codes_by_model: dict[str, dict[str, list[dict]]] = {}
    task_sets: list[set] = []
    for m in args.models:
        codes_path = os.path.join(args.evals_dir, m, "codes.jsonl")
        if not os.path.exists(codes_path):
            raise FileNotFoundError(f"缺少 {codes_path}")
        grouped: dict[str, list[dict]] = {}
        for c in _read_jsonl(codes_path):
            grouped.setdefault(c["task_id"], []).append(c)
        codes_by_model[m] = grouped
        task_sets.append(set(grouped.keys()))
        log.info(f"{m}: {sum(len(v) for v in grouped.values())} codes over {len(grouped)} problems")

    # 2) 选题:交集 → 排序固定顺序 → 抽 N(配对,所有模型同一批题)
    common = sorted(set.intersection(*task_sets))
    if len(common) < args.n_problems:
        log.warning(f"交集仅 {len(common)} 题 < 要求 {args.n_problems},全取交集")
    rng = random.Random(args.seed)
    n_take = min(args.n_problems, len(common))
    selected = sorted(rng.sample(common, n_take))
    log.info(f"共选 {len(selected)} 题(交集 {len(common)} 题)用于打分")

    # 3) 每模型每题随机封顶 M 条;写 sub/codes.jsonl + sub/exec.jsonl
    grand_total_calls = 0
    for m in args.models:
        grouped = codes_by_model[m]
        kept_codes: list[dict] = []
        kept_keys: set = set()
        for tid in selected:
            sols = grouped.get(tid, [])
            if not sols:
                continue
            if len(sols) > args.max_per_problem:
                tr = random.Random(_derive_seed(args.seed, m, tid))
                sols = tr.sample(sols, args.max_per_problem)
            for c in sols:
                kept_codes.append(c)
                kept_keys.add(_key(c))

        sub_dir = os.path.join(args.evals_dir, m, args.out_subdir)
        _write_jsonl(os.path.join(sub_dir, "codes.jsonl"), kept_codes)

        # 过滤对应 exec 记录(metrics 需要 pass_ratio/runtime;annotate 需要 pass_ratio 判阈值)
        exec_path = os.path.join(args.evals_dir, m, "exec.jsonl")
        kept_exec = [e for e in _read_jsonl(exec_path) if _key(e) in kept_keys]
        _write_jsonl(os.path.join(sub_dir, "exec.jsonl"), kept_exec)

        n_parsable = sum(1 for c in kept_codes if c.get("parsed_ok"))
        calls = n_parsable * 2  # 每条可评分解 = sketch + code 两次调用
        grand_total_calls += calls
        log.info(
            f"{m} → {args.out_subdir}/: {len(kept_codes)} codes "
            f"({n_parsable} parsable), {len(kept_exec)} exec, ~{calls} API calls"
        )

    log.info(f"全部模型合计 ~{grand_total_calls} 次 GLM-4-Air 调用(每条解 2 次)")
    log.info("下一步:对每个 {model}/%s 跑 04_annotate(pass_threshold=0.0)再跑 metrics" % args.out_subdir)


if __name__ == "__main__":
    main()
