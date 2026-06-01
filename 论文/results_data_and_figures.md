# 结果章节:数据归档 + 作图清单

> 归档时间 2026-06-01。所有数字来自 4090×2 云容器 `/root/shared-nvme/work/out/evals/`。
> 测试集:APPS interview test，2998 题。采样 temp=0.6。
> base/sft `n_per_temp≈100`，dpo 各任务 `n_per_temp=50`（pass@k 用无偏估计量，k≤n 时可比）。

---

## 0. 关键方法学前提（写图/写文都不能忘）

1. **DPO 从 SFT 初始化**：`07_train_dpo --model_path runs/sft_alg_top25/best`。
   ⇒ DPO 各任务的**正确参照系是 SFT**，不是 base。vs base 含 SFT 阶段增益，会掩盖 DPO 真实边际效果。
2. **循环论证 caveat**：rubric judge 与 GvB 偏好信号同源（均 GLM-4-Air 9 维 rubric）。
   rubric 提升不能单独作为质量证据，须与**独立执行指标**（compile_error 率、pass@1）绑讲。
3. **rubric 打分子采样**：400 题 × 每题 ≤10 解（配对，seed 1），`pass_threshold=0.0`，~4.8 万次调用。
   problem-level 聚合 + 问题层面 bootstrap（n_boot=10000）。

---

## 1. 现有数据全表

### 1a. pass@k / runtime（`summary.json`）

| model | pass@1 | pass@10 | mean_pass_ratio | median_rt | mean_rt | n_prob | n_sol_avg |
|---|---|---|---|---|---|---|---|
| base | 1.06% | 5.34% | 10.06% | 0.167ms | 9.862ms | 2998 | 99.67 |
| sft_alg_top25 | 1.27% | 5.30% | 11.73% | — | — | 2998 | 100.0 |
| dpo_pvf | 1.09% | 5.27% | 10.71% | — | — | 2998 | 50.0 |
| dpo_qvs | 0.57% | 3.11% | 7.88% | 0.177ms | 8.590ms | 2998 | 50.0 |
| dpo_gvb | 1.05% | 4.72% | 11.54% | 0.236ms | 20.158ms | 2998 | 50.0 |
| dpo_all | **1.27%** | **5.58%** | **11.96%** | 0.190ms | 10.412ms | 2998 | 50.0 |

⚠ **runtime 覆盖不齐**：sft / pvf 无 median_runtime（无 `--do_timing` 或无全通解可测）。多目标 Pareto 图需要 pvf 的 runtime → 见 §3 待补。

### 1b. algo_final / sketch / code（problem-level 均值 + 95% CI，`algo_summary.json`）

| model | algo_final | sketch | code |
|---|---|---|---|
| base | 3.723 [3.622,3.824] | 2.776 [2.70,2.85] | 4.354 [4.22,4.49] |
| sft_alg_top25 | 4.122 [4.015,4.230] | 3.389 [3.31,3.47] | 4.611 [4.47,4.75] |
| dpo_pvf | 3.820 [3.720,3.925] | 3.288 [3.21,3.37] | 4.175 [4.04,4.30] |
| dpo_qvs | 3.349 [3.250,3.451] | 2.658 [2.58,2.74] | 3.810 [3.69,3.94] |
| dpo_gvb | 4.141 [4.039,4.247] | 3.493 [3.41,3.58] | 4.573 [4.44,4.71] |
| dpo_all | 3.944 [3.839,4.053] | 3.303 [3.22,3.39] | 4.371 [4.23,4.51] |

### 1c. 9 维均值（S1–S4 sketch，C1–C5 code）

| model | S1 | S2 | S3 | S4 | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|---|---|---|
| base | 2.80 | 3.44 | 3.95 | 0.92 | 2.42 | 3.48 | 5.70 | 6.50 | 3.67 |
| sft_alg_top25 | 2.79 | 3.81 | 5.42 | 1.53 | 2.60 | 3.71 | 5.81 | 6.86 | 4.07 |
| dpo_pvf | 2.95 | 3.91 | 4.84 | 1.46 | 2.68 | 3.29 | 4.69 | 6.39 | 3.83 |
| dpo_qvs | 2.66 | 3.63 | 3.02 | 1.32 | 2.23 | 3.05 | 4.40 | 5.99 | 3.39 |
| dpo_gvb | 2.88 | 3.92 | 5.32 | 1.86 | 2.62 | 3.82 | 5.45 | 6.82 | 4.15 |
| dpo_all | 2.93 | 3.90 | 4.96 | 1.42 | 2.66 | 3.54 | 5.15 | 6.60 | 3.91 |

维度名：S1 correctness / S2 specificity / S3 complexity_awareness / S4 edge_coverage；
C1 faithfulness / C2 time_complexity / C3 space_complexity / C4 readability / C5 edge_handling。
**移动集中点**：S3（复杂度意识）、S4（边界覆盖）。qvs S3 暴跌 3.02；gvb S4=1.86 全场最高。

### 1d. 配对差分 vs base（`algo_summary.json` → paired_vs_baseline）

| model | Δ | 95% CI | p_boot | sig |
|---|---|---|---|---|
| sft_alg_top25 | +0.400 | [+0.338,+0.460] | 0.0000 | ✓ |
| dpo_pvf | +0.098 | [+0.038,+0.158] | 0.0012 | ✓ |
| dpo_qvs | −0.373 | [−0.440,−0.309] | 0.0000 | ✓ |
| dpo_gvb | +0.418 | [+0.352,+0.485] | 0.0000 | ✓ |
| dpo_all | +0.221 | [+0.160,+0.284] | 0.0000 | ✓ |

### 1e. 配对差分 vs SFT（**headline**，`algo_summary_vs_sft.json`）

| model | Δ | 95% CI | p_boot | sig |
|---|---|---|---|---|
| dpo_pvf | −0.302 | [−0.364,−0.242] | <0.001 | ✓ 回落 |
| dpo_qvs | −0.773 | [−0.842,−0.707] | <0.001 | ✓ 大幅回落 |
| **dpo_gvb** | **+0.019** | **[−0.036,+0.075]** | **0.4878** | **✗ 唯一不掉质量** |
| dpo_all | −0.179 | [−0.234,−0.123] | <0.001 | ✓ 回落 |

### 1f. 错误类型（来自此前 memory，仅 4 个 DPO 模型 + 2 类）

| | dpo_pvf | dpo_qvs | dpo_gvb | dpo_all |
|---|---|---|---|---|
| compile_error | 2.5% | 6.1% | **0.9%** ★最低 | 2.0% |
| hard_wall_timeout | 7.1% | 2.7% | 3.1% | **0.5%** ★最低 |

---

## 2. 要做的图（按论文重要性排序）

> 作图脚本全部在 `论文/figures/`，纯黑白、中文轴标题+图例、图例统一放画布右侧外部不压元素。
> 每个脚本顶部 dict 内嵌数据并带 §来源注释，改数据→重跑即可；输出 PDF（投稿矢量）+ PNG（预览）300dpi。
> 一键全跑见 §4。数据均已就位（除 Fig D 的 pvf runtime 待回填，见 §3）。

### 图 A（主图）— vs-SFT forest plot　`fig1_forest_vs_sft.py`
- **讲什么**：SFT 之后，只有 GvB 不显著掉质量；其余 DPO 目标质量回落。
- **图型**：横向 forest plot。y 轴 4 个 DPO 任务，x 轴 Δalgo_final，点 + 95% CI 横条，x=0 竖虚线。
- **数据**：`algo_summary_vs_sft.json` → `paired_vs_baseline[model]` 的 `mean_diff`、`ci95`（已嵌入 §1e）。
- **看点**：GvB 的 CI 跨 0（空心点）、其余全在左侧（实心点）。

### 图 B — 9 维分解（拆成两张独立图）　`fig2_dims.py`
- **讲什么**：质量提升集中在算法维度（S3 复杂度意识、S4 边界覆盖、C2 时间复杂度），非表层可读性。
- **图型**：分组柱状，**全 6 个模型**，按草图/代码拆两张独立图；6 模型用「4 级灰度 + 2 种白底纹理」区分。
  - **图 B1 `fig2a_sketch`** — 草图维度 S1–S4（ylim 0–6）
  - **图 B2 `fig2b_code`** — 代码维度 C1–C5（ylim 0–7.5）
  - 两图各自独立 y 轴缩放，草图维度（数值偏低）的 S3/S4 差距更醒目。
- **数据**：§1c 表（已嵌入脚本 `DATA`）。

### 图 C（缝合图）— algo_final × pass@1 散点　`fig3_quality_vs_pass.py`
- **讲什么**：把"质量"和"正确率"两个结论缝起来——GvB 在「高质量且不掉正确率」象限，QvS 在双输象限。直接回应循环论证 caveat。
- **图型**：散点，每模型一点（6 种点形状区分）；x=algo_final，y=pass@1；画 SFT 取值的十字虚线作参照系。
- **数据**：x 来自 §1b 的 `algo_final` 均值；y 来自 §1a 的 pass@1（已嵌入）。

### 图 D（多目标节）— median_runtime × pass@1 Pareto　`fig4_pareto_runtime.py`
- **讲什么**：多目标权衡——all 在 Pareto 前沿（pass@1 最高 + runtime 接近最快者）；qvs 拿正确率换速度被支配。
- **图型**：散点（点形状区分）+ Pareto 前沿黑色虚折线（脚本自动算前沿）；x=median_runtime（用 median，避免离群拖累），y=pass@1。
- **数据**：§1a 的 pass@1 + median_runtime。⚠ **pvf 的 runtime 待回填**（见 §3）；脚本里 `POINTS["PvF"]=None` 会自动跳过并提示，回填后填数值即自动重算前沿。

### 图 E（机制）— failure-mode 分组柱状　`fig5_failure.py`
- **讲什么**：GvB compile_error 最低、all hard_wall_timeout 最低 → 质量信号的执行层证据（抗循环论证）。
- **图型**：分组柱状，x=失败类型，4 个 DPO 模型用 灰度+纹理区分，柱顶标数值。
- **数据**：§1f 表（已嵌入）。⚠ 目前只有 4 DPO 模型 + 2 类错误；要扩到全分类 + base/sft 对照需重算（见 §3），脚本扩 `CATEGORIES`/`DATA` 即可。

---

## 3. 还缺 / 可能要补的数据

| 缺口 | 为什么要 | 怎么补 |
|---|---|---|
| **pvf 的 runtime（必补）** | 图 D Pareto 需要全部 4 个 DPO 模型的 median_runtime；pvf 现为空 | `backfill_timing` 复用 codes.jsonl 重跑执行 `--do_timing`（命令见下） |
| **sft 的 runtime（可选）** | 图 D 只画 4 个 DPO 模型，sft 不在 Pareto 上；仅当正文想顺嘴比 SFT 速度时才补 | 同 pvf，模型名换 `sft_alg_top25`；n≈100 ⇒ 重执行 ~30 万条，约 pvf 两倍慢 |
| **pass@1 的显著性（bootstrap CI）** | 现仅点估计；dpo_all(1.27) vs base(1.06) 差异极小，需 CI 才能说"是否真显著"。很可能不显著 → 更要靠 algo_final 立论 | metrics.py 不算 CI；需新增 problem-level bootstrap（可复用 analyze_algo_scores 的 boot 逻辑，对 per-problem pass 指标重采样） |
| **failure-mode 完整分类 + base/sft** | 图 E 现只 2 类、缺 base/sft 对照 | 从各模型 exec.jsonl 的 error 字段重新聚合全分类 |
| **pass@100（base/sft）** | 这两份 n≈100，可报 pass@100 增列；非必需 | `metrics.py --exec_path ...` ks 含 100 |
| **rubric 的独立验证** | 抗循环论证最强的是人工/异源 judge 抽验一小批 | 可选：抽 30–50 条让人工或换模型打分，报与 GLM-4-Air 的相关性。多半留作 limitation |
| **定性样例** | 附录放 1–2 个 GvB vs PvF 的 (sketch, code) 对照，直观展示算法考量差异 | 从 sub/codes.jsonl 挑同题不同模型的解 |

### backfill runtime 命令（pvf 必补 / sft 可选，同一套）

```bash
cd /root/shared-nvme/code/sketch-guided-clm-alignment
conda activate /root/shared-nvme/conda/envs/sketch4090
source v2/scripts/env_4090.sh                     # execution 沙箱依赖 Linux + SIGALRM
EVALS=/root/shared-nvme/work/out/evals
TEST=/root/shared-nvme/work/out/apps/test.jsonl
M=dpo_pvf                                          # sft 时改成 sft_alg_top25

# 1) 备份旧 exec(无 timing)——必须，否则 run_executions_parallel 断点续跑会 dedup 跳过全部
mv $EVALS/$M/exec.jsonl $EVALS/$M/exec.jsonl.notime.bak
# 2) 复用 codes.jsonl 重跑执行，开 timing（不占 GPU、不重新采样）
python -m v2.scripts.backfill_timing --eval_dir $EVALS/$M --problems_jsonl $TEST --exec_workers 24
# 3) 刷新汇总(pvf/sft 都补完后跑一次即可，幂等)
python -m v2.scripts.summarize_evals --evals_dir $EVALS --out $EVALS/summary.json
```

- 只有 `pass_ratio==1.0` 的解才进 timing（反复测到 CoV≤0.1）；pass/fail 确定性 ⇒ 重执行后 pass@1 应不变，仅新增 runtime。
- 备份文件名 `exec.jsonl.notime.bak` 与 base 之前的备份命名保持一致。
- 回填后核对 pass@1 无回归：pvf 仍应 ≈1.09%，sft 仍应 ≈1.27%。

---

## 4. 脚本与产物路径

- 子采样：`v2/scripts/subsample_for_scoring.py` → `evals/{model}/sub/{codes,exec}.jsonl`
- 打分：`python -m v2.scripts.04_annotate --sample_dir evals/{model}/sub --pass_threshold 0.0` → `sub/scores.jsonl`
- 分析：`v2/scripts/analyze_algo_scores.py` → `evals/algo_summary.json`（vs base）、`evals/algo_summary_vs_sft.json`（vs sft）
- pass@k 汇总：`v2/scripts/summarize_evals.py` → `evals/summary.json`
- 代码分支：`feat/4090-adaptation`（含上述两个新脚本，已 push）

**作图脚本（`论文/figures/`，本地 Windows 跑）：**
- 图 A：`fig1_forest_vs_sft.py` → `fig1_forest_vs_sft.{pdf,png}`
- 图 B1/B2：`fig2_dims.py` → `fig2a_sketch.{pdf,png}` + `fig2b_code.{pdf,png}`
- 图 C：`fig3_quality_vs_pass.py` → `fig3_quality_vs_pass.{pdf,png}`
- 图 D：`fig4_pareto_runtime.py` → `fig4_pareto_runtime.{pdf,png}`
- 图 E：`fig5_failure.py` → `fig5_failure.{pdf,png}`
- 一键全跑：`python make_all.py`（依次跑上面五个脚本）

## 5. 结果章节叙事骨架

1. 质量信号能注入（vs base，§1d）→ 除 QvS 外都显著抬高。
2. **headline**：DPO 从 SFT 初始化，vs SFT 只有 GvB 不掉质量（§1e，图 A）。
3. 与正确率联读：GvB 保质量且 pass@1 不降（§1a + 图 C）。
4. 机制：移动集中在 S3/S4/C2 算法维度（§1c，图 B）；执行层 compile_error 最低（§1f，图 E）。
5. 多目标权衡：all 在 Pareto 前沿（图 D）。
6. Caveat 段：循环论证 + 执行指标佐证。
