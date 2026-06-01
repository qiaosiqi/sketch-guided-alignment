# -*- coding: utf-8 -*-
"""
图 F：各模型运行时间分布（中位 median ↔ 90 分位 p90 区间棒棒糖图）。
讲什么：runtime 仅在 pass_ratio==1.0 的解上测量（反复测到 CoV≤0.1）；
        中位反映典型耗时、p90 反映长尾。QvS/All 最快，GvB 长尾略高。

数据来源：results_data_and_figures.md §1a（median_rt）+ memory（p90）。
全黑白：median 用空心圆、p90 用实心方，竖线相连；对数 y 轴。
图例放右侧外部，不覆盖元素。

⚠ 占位：SFT、PvF 的 runtime 待回填（backfill_timing，见 §3）；base 暂无 p90。
   跑出数字后把对应 None 换成数值即可，缺测项会自动跳过并在底部标“待回填”。

Windows 运行：
    python fig6_runtime.py
产出：fig6_runtime.pdf / .png
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

# ---- 数据：name -> (median_ms, p90_ms)；None = 待回填 ----
# median 来自 §1a；p90 来自 memory（目前仅 3 个 DPO 模型）。
RUNTIME = {
    "base": (0.167, None),
    "SFT":  (None,  None),   # ⚠ 待回填
    "PvF":  (None,  None),   # ⚠ 待回填（必补，见 §3）
    "QvS":  (0.177, 2.23),
    "GvB":  (0.236, 3.01),
    "All":  (0.190, 2.38),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    models = list(RUNTIME.keys())
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    drawn_median = drawn_p90 = False
    for i, m in enumerate(models):
        med, p90 = RUNTIME[m]
        if med is not None and p90 is not None:
            ax.plot([i, i], [med, p90], color="black", linewidth=1.2, zorder=2)
        if p90 is not None:
            ax.scatter(i, p90, marker="s", s=70, facecolor="black",
                       edgecolor="black", linewidth=0.8, zorder=3,
                       label="90 分位 p90" if not drawn_p90 else None)
            drawn_p90 = True
            ax.annotate(f"{p90:.2f}", xy=(i, p90), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
        if med is not None:
            ax.scatter(i, med, marker="o", s=70, facecolor="white",
                       edgecolor="black", linewidth=1.2, zorder=4,
                       label="中位 median" if not drawn_median else None)
            drawn_median = True
            ax.annotate(f"{med:.3f}", xy=(i, med), xytext=(0, -7),
                        textcoords="offset points", ha="center", va="top", fontsize=8.5)
        if med is None and p90 is None:
            ax.annotate("待回填", xy=(i, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color="0.45")

    ax.set_yscale("log")
    ax.set_ylim(0.12, 4.5)  # 留出底部 median 标签 / 顶部 p90 标签空间
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_xlabel("模型")
    ax.set_ylabel("运行时间（ms，对数轴）")
    ax.grid(axis="y", which="both", linestyle=":", color="0.7", linewidth=0.6, zorder=0)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fontsize=9.5, borderaxespad=0.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(args.outdir, f"fig6_runtime.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
