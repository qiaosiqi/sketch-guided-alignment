# -*- coding: utf-8 -*-
"""
图 B：rubric 维度均值分组柱状，全 6 个模型。
拆成两张独立的图：
    fig2a_sketch.*  —— 草图维度 S1–S4
    fig2b_code.*    —— 代码维度 C1–C5
讲什么：质量移动集中在 S3（复杂度意识）与 S4（边界覆盖）等算法维度，而非表层；
        QvS 的 S3 暴跌至 3.02，GvB 的 S4=1.86 全场最高。

数据来源：results_data_and_figures.md §1c。
全黑白：6 个模型用「4 级灰度 + 2 种白底纹理」区分，不用颜色。
图例放右侧外部，不覆盖柱子。

Windows 运行：
    python fig2_dims.py
产出：fig2a_sketch.pdf/.png、fig2b_code.pdf/.png
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Microsoft JhengHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "savefig.dpi": 300,
})

DIMS = ["S1", "S2", "S3", "S4", "C1", "C2", "C3", "C4", "C5"]

# ---- 数据：name -> 9 维均值（§1c，顺序同 DIMS）----
DATA = {
    "base": [2.80, 3.44, 3.95, 0.92, 2.42, 3.48, 5.70, 6.50, 3.67],
    "SFT":  [2.79, 3.81, 5.42, 1.53, 2.60, 3.71, 5.81, 6.86, 4.07],
    "PvF":  [2.95, 3.91, 4.84, 1.46, 2.68, 3.29, 4.69, 6.39, 3.83],
    "QvS":  [2.66, 3.63, 3.02, 1.32, 2.23, 3.05, 4.40, 5.99, 3.39],
    "GvB":  [2.88, 3.92, 5.32, 1.86, 2.62, 3.82, 5.45, 6.82, 4.15],
    "All":  [2.93, 3.90, 4.96, 1.42, 2.66, 3.54, 5.15, 6.60, 3.91],
}
# 6 模型黑白样式：4 级灰度 + 2 种白底纹理（黑白可分辨）
STYLE = {
    "base": dict(facecolor="white", hatch=""),
    "SFT":  dict(facecolor="0.72", hatch=""),
    "PvF":  dict(facecolor="white", hatch="///"),
    "QvS":  dict(facecolor="0.45", hatch=""),
    "GvB":  dict(facecolor="black", hatch=""),
    "All":  dict(facecolor="white", hatch="..."),
}


def plot_subset(dim_idx, ylim, fname, outdir):
    """对指定维度下标子集画一张分组柱状图。"""
    models = list(DATA.keys())
    n_model = len(models)
    labels = [DIMS[i] for i in dim_idx]
    x = np.arange(len(dim_idx))
    width = 0.8 / n_model

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for i, m in enumerate(models):
        offs = (i - (n_model - 1) / 2) * width
        vals = [DATA[m][j] for j in dim_idx]
        ax.bar(x + offs, vals, width=width, edgecolor="black",
               linewidth=0.8, label=m, zorder=3, **STYLE[m])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("评分维度")
    ax.set_ylabel("评分维度均值（0–10）")
    ax.grid(axis="y", linestyle=":", color="0.7", linewidth=0.6, zorder=0)

    ax.legend(title="模型", loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fontsize=9.5, borderaxespad=0.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(outdir, f"{fname}.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    # 草图维度 S1–S4（下标 0–3），代码维度 C1–C5（下标 4–8）
    plot_subset([0, 1, 2, 3], ylim=6.0, fname="fig2a_sketch", outdir=args.outdir)
    plot_subset([4, 5, 6, 7, 8], ylim=7.5, fname="fig2b_code", outdir=args.outdir)


if __name__ == "__main__":
    main()
