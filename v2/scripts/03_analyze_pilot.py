"""
Pilot 结果分析。读 02_sample_pilot.py 产生的 sketches/codes/exec.jsonl,
按温度 / 难度 切片,计算四个指标:

    1. parse_rate (sketch / code 各自解析成功率)
    2. unique_rate (sketch 去重率,code 去重率)
    3. survival_rate (能凑出 ≥2 全 pass + ≥1 全 fail 的题占比)
    4. pass_ratio 分布(直方图)
    5. gvb_yieldable (能在 pass_ratio ≥ θ_pass_gvb 内分出 >=1 个 G 和 >=1 个 B 的题数)
       — 注意这里只能用 pass_ratio 代理,因为 pilot 没跑 judge

按这四个指标和 README 的阈值规则给出推荐配置。

用法:
    python -m v2.scripts.03_analyze_pilot --pilot_dir out/pilot
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_dir", required=True)
    ap.add_argument("--theta_pass_gvb", type=float, default=0.8)
    args = ap.parse_args()

    sketches = load_jsonl(os.path.join(args.pilot_dir, "sketches.jsonl"))
    codes = load_jsonl(os.path.join(args.pilot_dir, "codes.jsonl"))
    execs = load_jsonl(os.path.join(args.pilot_dir, "exec.jsonl"))
    chosen = load_jsonl(os.path.join(args.pilot_dir, "chosen_problems.jsonl"))
    chosen_by_id = {p["task_id"]: p for p in chosen}

    # ---------- 1. parse_rate ----------
    print("=" * 60)
    print("1. Parse rate")
    print("=" * 60)
    by_temp_sketch = defaultdict(lambda: [0, 0])  # [ok, total]
    for s in sketches:
        t = s["sample_temp"]
        by_temp_sketch[t][1] += 1
        if s["parsed_ok"]:
            by_temp_sketch[t][0] += 1
    for t in sorted(by_temp_sketch):
        ok, tot = by_temp_sketch[t]
        print(f"  sketch @ T={t}: {ok}/{tot} = {ok/tot:.1%}")

    code_ok = sum(1 for c in codes if c["parsed_ok"])
    print(f"  code (all): {code_ok}/{len(codes)} = {code_ok/max(1,len(codes)):.1%}")

    # ---------- 2. unique_rate ----------
    print("\n" + "=" * 60)
    print("2. Unique rate (per problem)")
    print("=" * 60)
    sketches_by_pt = defaultdict(list)   # (task_id, temp) → sketch texts
    for s in sketches:
        if s["parsed_ok"]:
            sketches_by_pt[(s["task_id"], s["sample_temp"])].append(s["sketch"].strip())
    codes_by_p = defaultdict(list)
    for c in codes:
        if c["parsed_ok"]:
            codes_by_p[c["task_id"]].append(c["code"].strip())

    sketch_unique_by_temp = defaultdict(list)
    for (tid, temp), lst in sketches_by_pt.items():
        if lst:
            sketch_unique_by_temp[temp].append(len(set(lst)) / len(lst))
    for t in sorted(sketch_unique_by_temp):
        vals = sketch_unique_by_temp[t]
        print(f"  sketch unique_rate @ T={t}: mean={sum(vals)/len(vals):.2%} (n={len(vals)})")

    code_unique = []
    for tid, lst in codes_by_p.items():
        if lst:
            code_unique.append(len(set(lst)) / len(lst))
    if code_unique:
        print(f"  code unique_rate (all temps mixed): mean={sum(code_unique)/len(code_unique):.2%} "
              f"(n={len(code_unique)})")

    # ---------- 3. survival_rate (binary) ----------
    print("\n" + "=" * 60)
    print("3. Survival rate (binary pass/fail, like Code-Optimise)")
    print("=" * 60)
    exec_by_task = defaultdict(list)
    for e in execs:
        exec_by_task[e["task_id"]].append(e)

    survived_binary = 0
    survived_partial = 0
    for tid, es in exec_by_task.items():
        n_full_pass = sum(1 for e in es if e["pass_ratio"] == 1.0)
        n_full_fail = sum(1 for e in es if e["pass_ratio"] == 0.0)
        if n_full_pass >= 2 and n_full_fail >= 1:
            survived_binary += 1
        # partial-credit 版:有 ≥2 解 pass_ratio≥0.7 且 ≥1 解 pass_ratio≤0.3
        n_hi = sum(1 for e in es if e["pass_ratio"] >= 0.7)
        n_lo = sum(1 for e in es if e["pass_ratio"] <= 0.3)
        if n_hi >= 2 and n_lo >= 1:
            survived_partial += 1
    n_total = len(chosen_by_id)
    print(f"  binary survived (≥2 pass=1.0 + ≥1 pass=0.0): {survived_binary}/{n_total} = "
          f"{survived_binary/max(1,n_total):.1%}")
    print(f"  partial survived (≥2 pass≥0.7 + ≥1 pass≤0.3): {survived_partial}/{n_total} = "
          f"{survived_partial/max(1,n_total):.1%}")

    # ---------- 4. pass_ratio 分布 ----------
    print("\n" + "=" * 60)
    print("4. pass_ratio histogram (bins of 0.1)")
    print("=" * 60)
    bins = [0] * 11
    for e in execs:
        idx = min(10, int(e["pass_ratio"] * 10))
        bins[idx] += 1
    total = sum(bins) or 1
    for i, c in enumerate(bins):
        lo = i / 10
        hi = (i + 1) / 10 if i < 10 else 1.0
        bar = "#" * int(c / total * 50)
        print(f"  [{lo:.1f}, {hi:.1f}{']' if i==10 else ')'}: {c:5d} {bar}")

    # 按难度分别看
    print("\n  by difficulty:")
    diff_pass = defaultdict(list)
    for e in execs:
        diff = chosen_by_id.get(e["task_id"], {}).get("difficulty", "?")
        diff_pass[diff].append(e["pass_ratio"])
    for d, lst in diff_pass.items():
        if lst:
            zero = sum(1 for x in lst if x == 0.0)
            one = sum(1 for x in lst if x == 1.0)
            avg = sum(lst) / len(lst)
            print(f"    {d}: n={len(lst)} mean={avg:.3f} zeros={zero/len(lst):.1%} ones={one/len(lst):.1%}")

    # ---------- 5. GvB-yieldable (基于 pass_ratio 代理) ----------
    print("\n" + "=" * 60)
    print(f"5. GvB-yieldable (≥1 解 pass_ratio≥{args.theta_pass_gvb} AND ≥2 满足该阈值的解,"
          f"可在 judge 阶段分 G/B)")
    print("=" * 60)
    yieldable = 0
    for tid, es in exec_by_task.items():
        eligible = [e for e in es if e["pass_ratio"] >= args.theta_pass_gvb]
        if len(eligible) >= 2:
            yieldable += 1
    print(f"  yieldable: {yieldable}/{n_total} = {yieldable/max(1,n_total):.1%}")

    # ---------- 推荐 ----------
    print("\n" + "=" * 60)
    print("Recommendation")
    print("=" * 60)
    recs = []
    if sketch_unique_by_temp:
        best_temp = max(sketch_unique_by_temp.items(),
                        key=lambda kv: sum(kv[1])/len(kv[1]))
        recs.append(f"- Sketch 多样性最佳温度: {best_temp[0]} (unique={sum(best_temp[1])/len(best_temp[1]):.1%})")
    if survived_partial / max(1, n_total) < 0.5:
        recs.append("- survival_rate 偏低,建议加更低温度(增 pass 样本)或换更强 base model")
    if yieldable / max(1, n_total) < 0.3:
        recs.append(f"- GvB-yieldable 偏低,考虑降低 θ_pass_gvb (当前 {args.theta_pass_gvb})")
    if not recs:
        recs.append("- 各项指标健康,建议直接按 100/题、温度 0.6 全量采样")
    for r in recs:
        print(r)


if __name__ == "__main__":
    main()
