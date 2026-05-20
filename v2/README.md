# Sketch-Guided Multi-Objective Alignment v2

第二版框架,在原 `sketch-guided-alignment/` 论文方法基础上做以下扩展:

1. **数据集换 APPS Interview**(替代 MBPP);competition 难度因基础模型通过率近零、无法构造偏好对而排除
2. **Partial credit 主信号**:`pass_ratio ∈ [0,1]` 替代二元 pass/fail
3. **两段式采样,一段式训练**:采样时分别生成 sketch 和 code(各自评分更准),训练时拼成一段式格式
4. **多维度评分**:sketch 4 维 + code 5 维,各 0-10,加权求总分
5. **偏好对重定义**:HvL (High vs Low pass_ratio)、QvS (Quick vs Slow,仅在 pass_ratio=1.0 内比)、GvB (Good vs Bad algorithm score,需 pass_ratio ≥ 阈值);另保留 PvF (Pass vs Fail,二元 1.0/0.0) 作 HvL 的 ablation 对照
6. **技术栈升级**:trl/transformers/peft 当前稳定版,采样用 vllm

---

## 设计参数(默认值,可在 config 改)

| 项 | 值 | 备注 |
|---|---|---|
| Base model | `StarCoder2-3B` | 2 × RTX 5090-32G (Blackwell),bf16 全程 |
| Judge LLM | GLM-4-Air | 与原论文一致 |
| 采样数 | 100 / 题 | sketch + code 各一遍,共 200 次推理 |
| 采样温度 | 0.6 起步 | 多温度由 pilot 决定 |
| 输出长度 | 1024 tokens | 超 10% 截断再升 2048 |
| `θ_high` (HvL 高分阈值) | 0.7 | pass_ratio 高于此算 high |
| `θ_low` (HvL 低分阈值) | 0.3 | pass_ratio 低于此算 low |
| `θ_pass_gvb` | 0.5 | GvB 双方都得满足的最低 pass_ratio |
| `τ` (GvB 算法分阈值) | 6.0 | 总分 ≥ τ 为 Good,< τ 为 Bad |
| Sketch / Code 权重 | 0.4 / 0.6 | `final = 0.4·mean(S) + 0.6·mean(C)` |
| Train/Val split | 9:1 (在 APPS train 内) | 随机划分,固定 seed |
| Test split | APPS 官方 test 全量(过滤后) | |

---

## Pipeline

```
APPS raw
   │
   ▼ data/apps_loader.py          ──→ 统一 schema(题目 + IO 格式 + 测试用例)
   │
   ▼ sampling/sketch_sampler.py   ──→ 每题 N 个 sketch
   │
   ▼ sampling/code_sampler.py     ──→ 每个 sketch 配 1 个 code(或采多个)
   │
   ▼ execution/runner.py          ──→ 执行,得 pass_ratio + runtime(若全 pass)
   │
   ▼ annotation/judge_client.py   ──→ 两轮调 GLM-4-Air:评 sketch、评 code
   │
   ▼ merge/build_dataset.py       ──→ merged.jsonl (含 pass_ratio、runtime、9 维评分)
   │
   ▼ training/{pair_builder.py, train.py}
   │      ├─ SFT 模式:按 final score 取 top-p%
   │      └─ DPO 模式:HvL / QvS / GvB / ALL 偏好对
   │
   ▼ evaluation/eval_sampling.py  ──→ 同样的 sampling→execution→metrics,在 test 集上
```

---

## 偏好对正式定义(partial-credit 版)

对一道题的候选解集合 $\mathcal{Y}(x)$,每个 $y \in \mathcal{Y}$ 有 `(pass_ratio, runtime, algorithm_score)` 三元组。

- **HvL**:从 $\{y : \text{pass\_ratio} \geq \theta_\text{high}\}$ 中抽 $y_w$,从 $\{y : \text{pass\_ratio} \leq \theta_\text{low}\}$ 中抽 $y_l$
- **PvF**(ablation):从 $\{y : \text{pass\_ratio} = 1.0\}$ 中抽 $y_w$,从 $\{y : \text{pass\_ratio} = 0.0\}$ 中抽 $y_l$。不受任何阈值控制,作为 HvL 的二元退化对照
- **QvS**:从 $\{y : \text{pass\_ratio} = 1.0\}$ 中按 runtime 升序排,抽前段为 $y_w$、后段为 $y_l$
- **GvB**:从 $\{y : \text{pass\_ratio} \geq \theta_\text{pass\_gvb}\}$ 中按 algorithm_score 分 $\mathcal{G} = \{\cdot \geq \tau\}$、$\mathcal{B} = \{\cdot < \tau\}$,各抽一个

题目级过滤规则:对每类偏好对,只保留**能至少构造出一对**的题(否则该题对该信号无意义,跳过)。

---

## 目录

```
v2/
├── configs/      模型/数据/训练/评测的 YAML 配置
├── data/         APPS 加载、prompt 模板、splits
├── sampling/     sketch + code 两段式采样
├── execution/    stdin/stdout 和 函数调用 两种 runner
├── annotation/   GLM-4-Air judge + 9 维 rubric
├── merge/        汇总成训练用 merged.jsonl
├── training/     pair builder + SFT/DPO trainer
├── evaluation/   测试集采样 + 指标计算
└── scripts/      每个阶段的入口脚本(顺序编号)
```

---

## 状态

- [x] 设计文档 + 数据 schema + APPS loader + prompt 模板 (Phase 1)
- [x] Execution runners (Phase 2,Linux/macOS only)
- [x] Samplers + pilot (Phase 3,vllm 优先 / HF fallback)
- [x] Judge + merge (Phase 4,GLM-4-Air,9 维)
- [x] SFT (Phase 5a,trl 0.11+,DynamicSFTCollator)
- [x] DPO (Phase 5b,HvL/QvS/GvB/ALL,动态 pair 采样)
- [x] Evaluation (pass@k strict + mean_pass_ratio + runtime + algo_score)
- [ ] Sanity check + smoke tests

---

## 跑通顺序

```bash
# 0) 准备数据(读 APPS raw,过滤 + 9:1 拆分,落 jsonl)
python -m v2.scripts.01_prepare_apps \
    --apps_root /path/to/APPS/raw \
    --out_dir out/apps

# 1) Pilot(50 题 × 100 sketch × 3 温度 + 每 sketch 1 code + 执行;双卡 vllm DP)
bash v2/scripts/dp_sample.sh out/pilot \
    --problems_jsonl out/apps/train.jsonl \
    --model_path /path/to/StarCoder2-3B \
    --n_problems 50 --n_per_temp 100 --temps 0.4 0.7 1.0

# 2) 分析 pilot 结果,决定主跑配置
python -m v2.scripts.03_analyze_pilot --pilot_dir out/pilot

# 3) 全量主跑(配置依据 pilot 结果)
bash v2/scripts/dp_sample.sh out/main \
    --problems_jsonl out/apps/train.jsonl --model_path .../StarCoder2-3B \
    --n_problems 99999 --n_per_temp 100 --temps 0.6

# 4) GLM-4-Air 评分 (要 GLM_API_KEY 环境变量)
export GLM_API_KEY=...
python -m v2.scripts.04_annotate \
    --problems_jsonl out/apps/train.jsonl --sample_dir out/main \
    --pass_threshold 0.0

# 5) 合并成训练数据
python -m v2.scripts.05_merge \
    --problems_jsonl out/apps/train.jsonl --sample_dir out/main \
    --out out/datasets/train/merged.jsonl
# val 同理:用 val.jsonl + 单独的 sample_dir (val 集也要采样+评分+合并)

# 6a) SFT 训练 (5090 ×2, ZeRO-2 无 offload)
deepspeed --num_gpus 2 -m v2.scripts.06_train_sft \
    --train_merged out/datasets/train/merged.jsonl \
    --val_merged out/datasets/val/merged.jsonl \
    --model_path /path/to/StarCoder2-3B \
    --output_dir out/runs/sft_alg_top25 \
    --sort_by algo_final --top_p 25 --augment True \
    --ds_config v2/configs/ds_zero2_2gpu.json

# 6b) DPO 训练 (推荐从 SFT 检查点继续;ZeRO-3 切参更稳)
deepspeed --num_gpus 2 -m v2.scripts.07_train_dpo \
    --train_merged out/datasets/train/merged.jsonl \
    --val_merged out/datasets/val/merged.jsonl \
    --model_path out/runs/sft_alg_top25 \
    --output_dir out/runs/dpo_gvb \
    --task gvb --augment True \
    --ds_config v2/configs/ds_zero3_2gpu.json

# 7) 评测(test 集,双卡 vllm DP)
bash v2/scripts/dp_eval.sh out/evals/dpo_gvb \
    --problems_jsonl out/apps/test.jsonl \
    --model_path out/runs/dpo_gvb \
    --do_timing
```
