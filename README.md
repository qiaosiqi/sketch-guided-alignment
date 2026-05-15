# Sketch-Guided Multi-Objective Alignment for Code LMs

基于 [Code-Optimise (Gee et al., 2024)](https://arxiv.org/abs/2406.12502) 的扩展工作:在 `pass/fail` + `runtime` 两条偏好信号之外,引入 **算法质量(algorithm score)** 作为第三条正交信号,通过 LLM-as-Judge 的多维评分实现 sketch-guided 多目标对齐 DPO 训练。

本仓库包含两部分:

| 路径 | 角色 |
|---|---|
| `sketch-guided-alignment/` | 原论文(Code-Optimise + 论文初稿)的参考实现,**仅作对照**,不维护 |
| `v2/` | 重写的当前框架,目标 **APPS Competition + Interview**,**partial-credit pass_ratio** 主信号,**两段式 sketch+code 采样**,9 维评分,HvL / QvS / GvB DPO pair |
| `论文/` | 论文草稿 PDF |

**新论文方向**(在 v2 上做的工作,与参考论文的区别):
1. 数据集从 MBPP 换成 APPS Competition+Interview(更难,更适合考验算法质量信号)
2. 评测引入 `pass_ratio ∈ [0, 1]` 连续主信号,替代 binary pass/fail
3. 采样改两段式(先 sketch、后 code),sketch 与 code 分开评分
4. Judge rubric 改成 9 维细粒度(4 维 sketch + 5 维 code),加权聚合
5. DPO 偏好对在 partial-credit 下重定义(HvL 替代 PvF,QvS 仅在 pass_ratio=1.0 内,GvB 需 pass_ratio ≥ θ)

---

## 1. 快速部署(Linux 云主机,V100-32G)

```bash
# --- 0. 克隆 ---
git clone https://github.com/qiaosiqi/sketch-guided-alignment.git
cd sketch-guided-alignment

# --- 1. 装 Python 环境(conda 推荐) ---
conda create -n sketch python=3.10 -y
conda activate sketch

# 核心依赖(V100 → CUDA 12.1 fp16 路线)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.2 datasets==2.20.0 accelerate==0.34.2
pip install trl==0.11.4 peft==0.13.2 deepspeed==0.15.4
pip install vllm==0.6.3 numpy pandas requests tqdm tensorboard pytest

# --- 2. 下载 APPS 数据集 (1.3GB,git 已排除) ---
# 方式 A:从 Berkeley 官方 (推荐)
mkdir -p data && cd data
wget https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz
tar -xzf APPS.tar.gz   # 解出 APPS/raw/{train,test}/0000..4999/
cd ..

# 方式 B:从 HuggingFace
# python -c "from datasets import load_dataset; load_dataset('codeparrot/apps', split='train')"
# 然后自己写一段把 HF dataset 转回 APPS 原始目录格式

# --- 3. 下载 base model (StarCoder2-3B) ---
huggingface-cli download bigcode/starcoder2-3b --local-dir models/StarCoder2-3B

# --- 4. 配 GLM-4-Air API key (Phase 4 annotation 用) ---
export GLM_API_KEY="your_zhipuai_api_key"
# 永久化:写进 ~/.bashrc

# --- 5. 跑 smoke tests 验证环境 ---
pytest v2/tests -v
# 应该 87 passed (Linux 上 9 个 runner test 也能跑)
```

---

## 2. 跑实验出数据(全流程,按顺序)

每一步都有断点续跑,中断后重跑命令会从上次产物继续。

### 2.1 数据预处理(秒级)

```bash
python -m v2.scripts.01_prepare_apps \
    --apps_root data/APPS/raw \
    --out_dir out/apps
# 产物:out/apps/{train,val,test}.jsonl
# 过滤后 train ≈ 2200 题(Competition+Interview),val ≈ 240,test ≈ 2400
```

### 2.2 Pilot 验证(~30-60 分钟,vllm)

```bash
python -m v2.scripts.02_sample_pilot \
    --problems_jsonl out/apps/train.jsonl \
    --model_path models/StarCoder2-3B \
    --out_dir out/pilot \
    --n_problems 50 --n_per_temp 100 --temps 0.4 0.7 1.0

python -m v2.scripts.03_analyze_pilot --pilot_dir out/pilot
```

`03_analyze_pilot` 会报 5 个指标 + 给出温度推荐。**根据结果决定 2.3 用单温度还是多温度。**

### 2.3 全量采样(单卡 V100 vllm,~12-24 小时)

```bash
# train split
python -m v2.scripts.02_sample_pilot \
    --problems_jsonl out/apps/train.jsonl \
    --model_path models/StarCoder2-3B \
    --out_dir out/sample_train \
    --n_problems 99999 --n_per_temp 100 \
    --temps 0.6   # 或多温度,按 pilot 结论改

# val split 同理(改 problems_jsonl 和 out_dir)
python -m v2.scripts.02_sample_pilot \
    --problems_jsonl out/apps/val.jsonl \
    --model_path models/StarCoder2-3B \
    --out_dir out/sample_val \
    --n_problems 99999 --n_per_temp 100 --temps 0.6
```

### 2.4 GLM-4-Air 评分(API,~10-30 小时,取决于并发和阈值)

**关键参数 `--pass_threshold`**:只评 pass_ratio ≥ 此值的解,省 API 钱。
- `0.0`:全评(最贵,得到完整分布)
- `0.8`:推荐(只评接近 GvB 候选的解,API 调用量砍 60-80%)

```bash
python -m v2.scripts.04_annotate \
    --problems_jsonl out/apps/train.jsonl \
    --sample_dir out/sample_train \
    --pass_threshold 0.8 --alpha 0.4

python -m v2.scripts.04_annotate \
    --problems_jsonl out/apps/val.jsonl \
    --sample_dir out/sample_val \
    --pass_threshold 0.8 --alpha 0.4
```

API 调用粗算:`2360 题 × 100 解 × 2 次调用 ≈ 47 万次`(`pass_threshold=0`)。设 `0.8` 后约 8-15 万次。中断重跑会续。

### 2.5 合并训练数据(分钟级)

```bash
python -m v2.scripts.05_merge \
    --problems_jsonl out/apps/train.jsonl \
    --sample_dir out/sample_train \
    --out out/datasets/train/merged.jsonl

python -m v2.scripts.05_merge \
    --problems_jsonl out/apps/val.jsonl \
    --sample_dir out/sample_val \
    --out out/datasets/val/merged.jsonl
```

### 2.6 SFT 训练(~6-12 小时,DeepSpeed ZeRO-3 单卡 V100)

```bash
# 论文里的 ALG 风格(按算法分排 top-25%)
deepspeed --num_gpus 1 -m v2.scripts.06_train_sft \
    --train_merged out/datasets/train/merged.jsonl \
    --val_merged out/datasets/val/merged.jsonl \
    --model_path models/StarCoder2-3B \
    --output_dir out/runs/sft_alg_top25 \
    --sort_by algo_final --top_p 25 --augment True \
    --num_train_epochs 10 \
    --ds_config v2/configs/ds_zero3_offload.json

# 对照实验:SPD(按 runtime),PASS(按 pass_ratio)
# 只改 --sort_by 即可
```

### 2.7 DPO 训练(~6-12 小时,接 SFT 检查点继续)

```bash
# 主实验:GvB(本工作核心)
deepspeed --num_gpus 1 -m v2.scripts.07_train_dpo \
    --train_merged out/datasets/train/merged.jsonl \
    --val_merged out/datasets/val/merged.jsonl \
    --model_path out/runs/sft_alg_top25 \
    --output_dir out/runs/dpo_gvb \
    --task gvb --augment True \
    --ds_config v2/configs/ds_zero3_offload.json

# 对照:HvL / QvS / ALL —— 只改 --task 和 --output_dir
# 全参数 OOM 时加 --use_lora
```

### 2.8 评测(test 集,~6-12 小时)

```bash
python -m v2.scripts.08_eval \
    --problems_jsonl out/apps/test.jsonl \
    --model_path out/runs/dpo_gvb \
    --out_dir out/evals/dpo_gvb \
    --do_timing

# 同样跑 baseline / SFT-only / 其他 DPO 任务做对比
python -m v2.scripts.08_eval --problems_jsonl out/apps/test.jsonl \
    --model_path models/StarCoder2-3B --out_dir out/evals/base
python -m v2.scripts.08_eval --problems_jsonl out/apps/test.jsonl \
    --model_path out/runs/sft_alg_top25 --out_dir out/evals/sft_alg
# ... 以此类推
```

每个 `out/evals/*/metrics.json` 含 `pass@1/10/100`、`mean_pass_ratio`、`mean_runtime_ns`、`mean_algo_final`。

---

## 3. 实验对比表(论文用)

至少需要跑这些组,出一张主表:

| 实验 | 命令 |
|---|---|
| BASE | StarCoder2-3B,直接 eval |
| SFT-SPD-25 | `--sort_by runtime --top_p 25` |
| SFT-PASS-25 | `--sort_by pass_ratio --top_p 25` |
| SFT-ALG-25 | `--sort_by algo_final --top_p 25` |
| SFT-ALG-100 | `--sort_by algo_final --top_p 100` |
| DPO-HvL | `--task hvl`(从 SFT-ALG-25 继续,partial-credit 主信号) |
| DPO-PvF | `--task pvf`(binary 退化:1.0 vs 0.0,作 HvL 的 ablation) |
| DPO-QvS | `--task qvs` |
| DPO-GvB | `--task gvb` (主实验) |
| DPO-ALL | `--task all`(不含 pvf;pvf 只用作消融) |

---

## 4. 框架状态

- [x] Phase 1 — Data schema / prompts / APPS loader
- [x] Phase 2 — Execution (stdio + fncall,partial credit,稳定测时)
- [x] Phase 3 — Sampling(两段式,vllm + HF 双后端)
- [x] Phase 4 — Judge(GLM-4-Air + 9 维 rubric)+ Merge
- [x] Phase 5a — SFT(trl 0.11+,DynamicSFTCollator)
- [x] Phase 5b — DPO(HvL / QvS / GvB / ALL,动态 pair 采样)
- [x] Phase 5c — Evaluation(pass@k + mean_pass_ratio + runtime + algo)
- [x] Smoke tests — 87 个用例(78 Win + 9 Linux-only)

各 phase 的设计细节见 [`v2/README.md`](v2/README.md);测试细节见 [`v2/tests/README.md`](v2/tests/README.md);上层代码导览见 [`CLAUDE.md`](CLAUDE.md)。

---

## 5. 排错速查

| 症状 | 原因 / 解决 |
|---|---|
| `RuntimeError: no kernel image is available` (vllm) | V100 不支持某些算子,环境变量 `VLLM_ATTENTION_BACKEND=XFORMERS` 或回退 `--prefer_backend hf` |
| `bf16 not supported` | V100 不支持 bf16,确认 config 是 `fp16=True, bf16=False` |
| OOM at DPO | 加 `--use_lora`,或减 `--per_device_train_batch_size 1 --gradient_accumulation_steps 32` |
| 评分太慢 | 提高 `--pass_threshold 0.8`,或写并发 wrapper 提速 |
| `signal.SIGALRM` AttributeError | 在 Windows 上跑 execution → 必须用 Linux |
| 模型 import trust_remote_code 警告 | StarCoder2 需要 `trust_remote_code=True`,代码已设 |

---

## License

本项目部分代码基于 Code-Optimise (Apache 2.0),保留相应版权声明。
