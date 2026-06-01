# -*- coding: utf-8 -*-
"""
图 4（多目标）：中位运行时间 median runtime × 正确率 pass@1 的 Pareto 图。
讲什么：All 在 Pareto 前沿（pass@1 最高、runtime 接近最快者）；QvS 拿正确率换速度被支配。

数据来源：results_data_and_figures.md §1a（median_rt, pass@1）。
全黑白：不同点形状区分模型（无颜色）；Pareto 前沿用黑色虚折线连接。
图例放右侧外部，不覆盖任何点或前沿线。

⚠ PvF 的 median_runtime 待 backfill_timing 回填（§3）。回填后把 POINTS["PvF"] 的 None
   换成实测值即可，前沿会自动重算。

Windows 运行：
    python fig4_pareto_runtime.py
产出：fig4_pareto_runtime.pdf / .png
"""
import argparse
import os

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Microsoft JhengHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "savefig.dpi": 300,
})

# ---- 数据：name -> (median_runtime_ms[§1a], pass@1%[§1a], 点形状) ----
# 目标：x 越小越好、y 越大越好 ⇒ Pareto 最优在左上。
# runtime 为 None 表示缺测（不参与作图/前沿），回填后填入数值。
POINTS = {
    "base": (0.167, 1.06, "o"),   # 参照
    "PvF":  (None,  1.09, "^"),   # ⚠ 待回填 runtime
    "QvS":  (0.177, 0.57, "v"),
    "GvB":  (0.236, 1.05, "D"),
    "All":  (0.190, 1.27, "P"),
}


def pareto_frontier(items):
    """items: list[(x, y, name)]；最小化 x、最大化 y。返回前沿点（按 x 升序）。"""
    pts = sorted(items, key=lambda t: (t[0], -t[1]))
    frontier, best_y = [], float("-inf")
    for x, y, name in pts:
        if y > best_y:
            frontier.append((x, y, name))
            best_y = y
    return frontier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    drawn = [(x, y, name) for name, (x, y, _) in POINTS.items() if x is not None]

    # Pareto 前沿折线（先画，置于点下层）
    front = pareto_frontier(drawn)
    if len(front) >= 2:
        fx = [p[0] for p in front]
        fy = [p[1] for p in front]
        ax.plot(fx, fy, linestyle="--", color="black", linewidth=1.2,
                zorder=2, label="Pareto 前沿")

    for name, (x, y, mk) in POINTS.items():
        if x is None:
            continue
        ax.scatter(x, y, marker=mk, s=90, facecolor="black", edgecolor="black",
                   linewidth=0.8, zorder=3, label=name)

    ax.set_xlabel("中位运行时间  median runtime（ms，越小越好 ←）")
    ax.set_ylabel("pass@1（%，越大越好 ↑）")
    ax.set_title("正确率 × 运行效率的多目标权衡", fontsize=11)
    ax.grid(True, linestyle=":", color="0.7", linewidth=0.6, zorder=0)

    # 缺测模型在标题下方提示（不进图例，避免误导）
    missing = [n for n, (x, _, _) in POINTS.items() if x is None]
    if missing:
        ax.annotate("缺 runtime（待回填）：" + "、".join(missing),
                    xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 14),
                    textcoords="offset points", fontsize=8.5, color="0.4", va="bottom")

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fontsize=9, borderaxespad=0.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(args.outdir, f"fig4_pareto_runtime.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
