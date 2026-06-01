# -*- coding: utf-8 -*-
"""
图 3（缝合图）：算法质量 algo_final × 正确率 pass@1 散点。
讲什么：GvB 在保住质量（≈SFT）的同时不损 pass@1；QvS 双输（左下）。

数据来源：results_data_and_figures.md §1a（pass@1）+ §1b（algo_final）。
全黑白：每个模型用不同的点形状区分（无颜色）；以 SFT 取值画十字基准虚线，
标出「相对 SFT」的四象限。图例放右侧外部，不覆盖任何点。

Windows 运行：
    python fig3_quality_vs_pass.py
产出：fig3_quality_vs_pass.pdf / .png
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

# ---- 数据：name -> (algo_final[§1b], pass@1%[§1a], 点形状) ----
POINTS = {
    "base": (3.723, 1.06, "o"),
    "SFT":  (4.122, 1.27, "s"),
    "PvF":  (3.820, 1.09, "^"),
    "QvS":  (3.349, 0.57, "v"),
    "GvB":  (4.141, 1.05, "D"),
    "All":  (3.944, 1.27, "P"),
}
SFT_X, SFT_Y = POINTS["SFT"][0], POINTS["SFT"][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    # SFT 基准十字（参照系：DPO 从 SFT 初始化）
    ax.axvline(SFT_X, linestyle="--", color="0.5", linewidth=1.0, zorder=1)
    ax.axhline(SFT_Y, linestyle="--", color="0.5", linewidth=1.0, zorder=1)
    ax.annotate("SFT 基准", xy=(SFT_X, ax.get_ylim()[0]), xytext=(2, 2),
                textcoords="offset points", color="0.4", fontsize=9, va="bottom")

    for name, (x, y, mk) in POINTS.items():
        ax.scatter(x, y, marker=mk, s=90, facecolor="black", edgecolor="black",
                   linewidth=0.8, zorder=3, label=name)

    ax.set_xlabel("算法质量得分  algo_final（GLM-4-Air 9 维 rubric，0–10）")
    ax.set_ylabel("pass@1（%）")
    ax.set_title("算法质量 × 正确率（虚线为 SFT 基准）", fontsize=11)
    ax.grid(True, linestyle=":", color="0.7", linewidth=0.6, zorder=0)

    ax.legend(title="模型", loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fontsize=9, borderaxespad=0.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(args.outdir, f"fig3_quality_vs_pass.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
