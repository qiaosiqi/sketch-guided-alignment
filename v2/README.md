# Sketch-Guided Multi-Objective Alignment v2

第二版框架,在原 `sketch-guided-alignment/` 论文方法基础上做以下扩展:

1. **数据集换 APPS Interview**(替代 MBPP);competition 难度因基础模型通过率近零、无法构造偏好对而排除
2. **Pass-ratio 门控**:执行得 `pass_ratio ∈ [0,1]`,作为 QvS(仅 1.0 入选)、GvB(≥ θ_pass_gvb 入选)的入门条件,PvF 用极值 1.0 / 0.0 构造二元正确性对
3. **两段式采样,一段式训练**:采样时分别生成 sketch 和 code(各自评分更准),训练时拼成一段式格式
4. **多维度算法评分**:sketch 4 维 + code 5 维,各 0-10,加权求总分,为 **GvB** 偏好对提供独立于 pass_ratio 的"算法质量"信号
5. **多目标偏好对**:PvF (正确性主信号,1.0 vs 0.0)、QvS (Quick vs Slow,仅在 pass_ratio=1.0 内比 runtime)、GvB (Good vs Bad algorithm score,需 pass_ratio ≥ θ_pass_gvb)
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
   │      └─ DPO 模式:PvF / QvS / GvB / ALL 偏好对
   │
   ▼ evaluation/eval_sampling.py  ──→ 同样的 sampling→execution→metrics,在 test 集上
```

---

## 偏好对正式定义

对一道题的候选解集合 $\mathcal{Y}(x)$,每个 $y \in \mathcal{Y}$ 有 `(pass_ratio, runtime, algorithm_score)` 三元组。

- **PvF**(正确性主信号):从 $\{y : \text{pass\_ratio} = 1.0\}$ 中抽 $y_w$,从 $\{y : \text{pass\_ratio} = 0.0\}$ 中抽 $y_l$;不受任何阈值控制
- **QvS**:从 $\{y : \text{pass\_ratio} = 1.0\}$ 中按 runtime 升序排,抽前段为 $y_w$、后段为 $y_l$
- **GvB**(论文核心卖点):从 $\{y : \text{pass\_ratio} \geq \theta_\text{pass\_gvb}\}$ 中按 algorithm_score 分 $\mathcal{G} = \{\cdot \geq \tau\}$、$\mathcal{B} = \{\cdot < \tau\}$,各抽一个

题目级过滤规则:对每类偏好对,只保留**能至少构造出一对**的题(否则该题对该信号无意义,跳过)。

> 设计回溯(2026-05-21):原方案曾计划用 partial-credit HvL 作为主信号(`pass_ratio ≥ 0.7` vs `≤ 0.3`),但 pilot 实测 StarCoder2-3B × APPS interview 上 pass_ratio 分布近似二元(92% = 0,4.6% = 1,中间几乎为空),HvL 与 PvF 偏好对几乎相同,partial-credit 信号在该 setting 下无增量。故 HvL 整条信号通道下线,PvF 升格为唯一的"正确性"任务。

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
- [x] DPO (Phase 5b,PvF/QvS/GvB/ALL,动态 pair 采样)
- [x] Evaluation (pass@k strict + mean_pass_ratio + runtime + algo_score)
- [ ] Sanity check + smoke tests

---

## 环境部署(5090 ×2,Blackwell sm_120)

分阶段装,不用 yml。只 pin 三个不让步的版本(torch 2.7+cu128 整套、vllm ≥ 0.9、trl ≥ 0.11),其余交给 pip 反向约束。

```bash
bash v2/scripts/setup_env.sh
conda activate /data/conda/envs/sketch5090
```

环境默认装在 `/data/conda/envs/sketch5090`(根盘空间小);要改装别处,设 `ENV_PREFIX=/your/path` 后再跑。脚本会在每个关键步骤后核验(torch 是否拿到 sm_120、vllm 装完 torch 没被换掉、trl ≥ 0.11 的新 API 能否 import),失败立即退出。

---

## 前置准备(首次部署)

云主机布局:`/` 留给 OS,大文件全压 `/data`。下面四步只在新机器上做一次。

### 1. 缓存路径 export 进 `~/.bashrc`

不设这五个,HF 模型下载、torch JIT、deepspeed 编译都会落到 `/`,系统盘很快爆。

```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/data/cache/hf
export MODELSCOPE_CACHE=/data/cache/modelscope
export TRITON_CACHE_DIR=/data/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor
export TMPDIR=/data/tmp
EOF
mkdir -p /data/cache/{hf,modelscope,triton,torchinductor} /data/tmp
source ~/.bashrc
```

### 2. 下载 StarCoder2-3B(~7 GB)

```bash
# 选 A:HF(优先)
huggingface-cli download bigcode/starcoder2-3b \
    --local-dir /data/models/StarCoder2-3B --local-dir-use-symlinks False

# 选 B:HF 被墙时走 modelscope
python -c "from modelscope import snapshot_download; \
    snapshot_download('AI-ModelScope/starcoder2-3b', cache_dir='/data/models')"
```

### 3. 下载 APPS 原始数据(~1.3 GB)

`v2/data/apps_loader.py` 要的是 hendrycks 原版目录结构(`<root>/{train,test}/<id>/{question.txt, input_output.json, ...}`),不是 parquet 封装。

```bash
huggingface-cli download codeparrot/apps --repo-type dataset \
    --local-dir /data/datasets/APPS --local-dir-use-symlinks False
# 核验:应能看到一堆数字编号目录
ls /data/datasets/APPS/train | head -5
```

### 4. 生成 jsonl splits

```bash
python -m v2.scripts.01_prepare_apps \
    --apps_root /data/datasets/APPS \
    --out_dir /data/work/out/apps
```

产 `train.jsonl` / `val.jsonl` / `test.jsonl`(只 interview 难度,9:1 拆 train/val)。几秒,不动 GPU。

---

## 跑通顺序

下面所有 `--out_dir` / `--output_dir` 均建议放 `/data/work/out/...`,避免训练 ckpt 把系统盘塞爆。

```bash
# 1) Pilot(50 题 × 100 sketch × 3 温度 + 每 sketch 1 code + 执行;双卡 vllm DP)
# --exec_workers 可选(默认按 (cpu_count-4)/n_shards 推算,28 核双 shard → 每 shard 12)
bash v2/scripts/dp_sample.sh /data/work/out/pilot \
    --problems_jsonl /data/work/out/apps/train.jsonl \
    --model_path /data/models/StarCoder2-3B \
    --n_problems 50 --n_per_temp 100 --temps 0.4 0.7 1.0

# 2) 分析 pilot 结果,决定主跑配置
python -m v2.scripts.03_analyze_pilot --pilot_dir /data/work/out/pilot

# 3) 全量主跑(配置依据 pilot 结果)
# --do_timing 是 DPO-QvS 任务的必需输入,主跑必开;代价 execution 阶段慢 3-10× (稳定测时 CoV ≤ 0.1)
bash v2/scripts/dp_sample.sh /data/work/out/main \
    --problems_jsonl /data/work/out/apps/train.jsonl \
    --model_path /data/models/StarCoder2-3B \
    --n_problems 99999 --n_per_temp 100 --temps 0.6 --do_timing

# 4) GLM-4-Air 评分 (要 GLM_API_KEY 环境变量)
export GLM_API_KEY=...
python -m v2.scripts.04_annotate \
    --problems_jsonl /data/work/out/apps/train.jsonl \
    --sample_dir /data/work/out/main \
    --pass_threshold 0.0

# 5) 合并成训练数据
python -m v2.scripts.05_merge \
    --problems_jsonl /data/work/out/apps/train.jsonl \
    --sample_dir /data/work/out/main \
    --out /data/work/out/datasets/train/merged.jsonl
# val 同理:用 val.jsonl + 单独的 sample_dir (val 集也要采样+评分+合并)

# 6a) SFT 训练 (5090 ×2, ZeRO-3;ZeRO-2 在 Adam init 就 OOM)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --num_gpus 2 --module v2.scripts.06_train_sft \
    --train_merged /data/work/out/datasets/train/merged.jsonl \
    --val_merged /data/work/out/datasets/val/merged.jsonl \
    --model_path /data/models/StarCoder2-3B \
    --output_dir /data/work/out/runs/sft_alg_top25 \
    --sort_by algo_final --top_p 25 --augment True \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --ds_config v2/configs/ds_zero3_2gpu.json

# 6b) DPO 训练 (推荐从 SFT 检查点继续)
# 配置:ZeRO-2 + CPU offload optim + precompute_ref_log_probs
#   - TRL 禁止 ZeRO-3 + precompute,但不 precompute 又装不下 policy + ref 两个 3B
#   - 解法:ZeRO-2 把 12GB Adam state 甩到 CPU,腾出 GPU 给 policy + ref;precompute 后释放 ref
# --augment True 走 K-pair 静态扩展:每题预采 K=--pairs_per_problem(默认 = --num_train_epochs)
#   个 pair,trainer 内部把 epoch 数缩成 max(1, E//K),总训练量等价原 dynamic 模式
DS_SKIP_CUDA_CHECK=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --num_gpus 2 --module v2.scripts.07_train_dpo \
    --train_merged /data/work/out/datasets/train/merged.jsonl \
    --val_merged /data/work/out/datasets/val/merged.jsonl \
    --model_path /data/work/out/runs/sft_alg_top25/best \
    --output_dir /data/work/out/runs/dpo_gvb \
    --task gvb --augment True \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --ds_config v2/configs/ds_zero2_2gpu_offload.json

# 7) 评测(test 集,双卡 vllm DP)
bash v2/scripts/dp_eval.sh /data/work/out/evals/dpo_gvb \
    --problems_jsonl /data/work/out/apps/test.jsonl \
    --model_path /data/work/out/runs/dpo_gvb/best \
    --do_timing
```

> **`/best` 路径约定**:SFT / DPO 训练完后,`output_dir/best` 是符号链接,指向按 val_loss 排名第一的 epoch ckpt(由 `trainer.state.best_model_checkpoint` 标记)。所有下游命令(DPO 接 SFT、评测接 DPO)的 `--model_path` 都从 `/best` 进。
>
> 之所以采用 symlink 而非 `load_best_model_at_end=True`:5090 32GB + 3B 模型 + ZeRO-3 优化器状态 + reload 临时显存 → trainer 内部 reload 会 OOM。Symlink 等价于 `load_best=True` 的语义(都是选 val_loss 最低那个 ckpt),但只动文件系统不动 GPU。
