# -*- coding: utf-8 -*-
"""
图 1（主图）：各 DPO 任务相对 SFT 的算法质量变化 Δalgo_final —— 横向 forest plot。

数据来源：results_data_and_figures.md §1e（algo_summary_vs_sft.json 的 paired_vs_baseline）。
全黑白：用「实心点 = 显著回落（95% CI 不含 0）」「空心点 = 无显著差异（CI 含 0）」区分，不用颜色。
图例放画布右侧外部，不覆盖任何点或误差条。

Windows 运行：
    python fig1_forest_vs_sft.py
产出：同目录下 fig1_forest_vs_sft.pdf / .png
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- 字体 / 全局样式（中文 + 负号）----
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Microsoft JhengHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "savefig.dpi": 300,
})

# ---- 数据：(显示名, Δ, ci_lo, ci_hi, p_boot)  —— §1e，回填后改这里即可 ----
# significant 由 CI 是否含 0 自动判定，不手填。
ROWS = [
    ("GvB", +0.019, -0.036, +0.075, 0.488),   # 唯一 CI 含 0
    ("All", -0.179, -0.234, -0.123, 0.0),      # p<0.001
    ("PvF", -0.302, -0.364, -0.242, 0.0),      # p<0.001
    ("QvS", -0.773, -0.842, -0.707, 0.0),      # p<0.001
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    # 从上到下排列：列表第 0 项在最上方
    n = len(ROWS)
    ys = list(range(n - 1, -1, -1))  # 倒序，使第 0 行 y 最大（最上）

    for y, (name, mean, lo, hi, p) in zip(ys, ROWS):
        sig = not (lo <= 0 <= hi)          # CI 不含 0 ⇒ 显著
        xerr = [[mean - lo], [hi - mean]]  # 非对称误差条
        face = "black" if sig else "white"
        ax.errorbar(
            mean, y, xerr=xerr,
            fmt="o", markersize=8, markerfacecolor=face, markeredgecolor="black",
            ecolor="black", elinewidth=1.4, capsize=4, capthick=1.4, zorder=3,
        )
        # 在误差条上方标注 Δ 与 p
        p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.annotate(f"Δ={mean:+.3f}  {p_str}",
                    xy=(mean, y), xytext=(0, 11), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    # Δ=0 基准竖虚线
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0, zorder=1)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in ROWS])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("相对 SFT 的算法质量得分变化 ΔS_algo")
    ax.set_ylabel("DPO 偏好任务")
    ax.set_title("各 DPO 任务相对 SFT 的算法质量变化（95% 置信区间）", fontsize=11)
    ax.grid(axis="x", linestyle=":", color="0.6", linewidth=0.6, zorder=0)

    # ---- 图例（代理句柄），放右侧外部，不压点 ----
    handles = [
        Line2D([0], [0], marker="o", markersize=8, markerfacecolor="black",
               markeredgecolor="black", linestyle="None", label="回落（95% CI）"),
        Line2D([0], [0], marker="o", markersize=8, markerfacecolor="white",
               markeredgecolor="black", linestyle="None", label="提升（95% CI）"),
        Line2D([0], [0], color="black", linewidth=1.4, label="95% 置信区间"),
        Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", label="Δ=0（与 SFT 无差异）"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fontsize=9, borderaxespad=0.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(args.outdir, f"fig1_forest_vs_sft.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
