# -*- coding: utf-8 -*-
"""
图 E（机制）：失败类型占比分组柱状。
讲什么（执行层独立证据，抗 rubric 循环论证）：
    GvB 的 compile_error 最低（0.9%），All 的 hard_wall_timeout 最低（0.5%）；
    QvS 的 compile_error 翻倍（6.1%）——短代码更快但更易碎。

数据来源：results_data_and_figures.md §1f（目前仅 4 个 DPO 模型 + 2 类）。
全黑白：四个 DPO 模型用 灰度 + 斜纹/点纹填充 区分，不用颜色。柱顶标数值。
图例放右侧外部，不覆盖柱子。

⚠ §1f 仅 2 类、缺 base/sft 对照。若日后从各 exec.jsonl 重聚合全分类，
   把 CATEGORIES / DATA 扩展即可，作图逻辑无需改。

Windows 运行：
    python fig5_failure.py
产出：fig5_failure.pdf / .png
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

# x 轴分组：失败类型（显示名）
CATEGORIES = ["编译错误\ncompile_error", "硬超时\nhard_wall_timeout"]

# ---- 数据：model -> [各类型占比%]，顺序同 CATEGORIES（§1f）----
DATA = {
    "PvF": [2.5, 7.1],
    "QvS": [6.1, 2.7],
    "GvB": [0.9, 3.1],
    "All": [2.0, 0.5],
}
# 灰度 + 填充纹理（黑白可分辨）
STYLE = {
    "PvF": dict(facecolor="white", hatch="///"),
    "QvS": dict(facecolor="0.45", hatch=""),
    "GvB": dict(facecolor="black", hatch=""),
    "All": dict(facecolor="white", hatch="..."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    models = list(DATA.keys())
    n_model = len(models)
    x = np.arange(len(CATEGORIES))
    width = 0.8 / n_model

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for i, m in enumerate(models):
        offs = (i - (n_model - 1) / 2) * width
        bars = ax.bar(x + offs, DATA[m], width=width, edgecolor="black",
                      linewidth=0.8, label=m, zorder=3, **STYLE[m])
        for b, v in zip(bars, DATA[m]):
            ax.annotate(f"{v:.1f}", xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim(0, max(max(v) for v in DATA.values()) * 1.18)
    ax.set_xlabel("失败类型")
    ax.set_ylabel("占全部生成解的比例（%）")
    ax.set_title("失败类型分布（执行层证据：GvB 编译错误最低、All 硬超时最低）", fontsize=10)
    ax.grid(axis="y", linestyle=":", color="0.7", linewidth=0.6, zorder=0)

    ax.legend(title="模型", loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fontsize=9.5, borderaxespad=0.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(args.outdir, f"fig5_failure.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
