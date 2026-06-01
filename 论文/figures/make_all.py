# -*- coding: utf-8 -*-
"""
一键跑齐所有结果图脚本（论文/figures/ 下）。
Windows 运行：
    python make_all.py
依次执行 fig1 / fig2(→B1+B2) / fig3 / fig4 / fig5，产出 PDF + PNG。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "fig1_forest_vs_sft.py",
    "fig2_dims.py",          # 输出 fig2a_sketch + fig2b_code 两张
    "fig3_quality_vs_pass.py",
    "fig4_pareto_runtime.py",
    "fig5_failure.py",
]


def main():
    failed = []
    for s in SCRIPTS:
        path = os.path.join(HERE, s)
        print(f"\n=== {s} ===")
        rc = subprocess.run([sys.executable, path], cwd=HERE).returncode
        if rc != 0:
            failed.append(s)
    if failed:
        print("\n[FAILED]", ", ".join(failed))
        sys.exit(1)
    print("\n全部图已生成。")


if __name__ == "__main__":
    main()
