# Sketch-Guided Multi-Objective Alignment for Code LMs

基于 [Code-Optimise (Gee et al., 2024)](https://arxiv.org/abs/2406.12502) 的扩展工作:在 `pass/fail` + `runtime` 两条偏好信号之外,引入 **算法质量(algorithm score)** 作为第三条正交信号,通过 LLM-as-Judge 的多维评分实现 sketch-guided 多目标对齐 DPO 训练。

本仓库包含两部分:

| 路径 | 角色 |
|---|---|
| `sketch-guided-alignment/` | 原论文(Code-Optimise + 论文初稿)的参考实现,**仅作对照**,不维护 |
| `v2/` | 重写的当前框架,目标 **APPS Interview**,**两段式 sketch+code 采样**,9 维评分,**PvF / QvS / GvB DPO pair**(GvB 为论文核心卖点) |
| `论文/` | 论文草稿 PDF |

**新论文方向**(在 v2 上做的工作,与参考论文的区别):
1. 数据集从 MBPP 换成 APPS Interview(比 MBPP 难,更适合考验算法质量信号;competition 因通过率近零被排除)
2. 评测产 `pass_ratio ∈ [0, 1]`,作为 QvS / GvB 的门控阈值(QvS 仅 1.0 入选,GvB 需 ≥ θ_pass_gvb)
3. 采样改两段式(先 sketch、后 code),sketch 与 code 分开评分
4. Judge rubric 改成 9 维细粒度(4 维 sketch + 5 维 code),加权聚合,为 GvB 提供独立于 pass_ratio 的算法质量信号
5. DPO 走多目标偏好对:**PvF**(二元正确性,1.0 vs 0.0)、**QvS**(runtime,全 pass 内比)、**GvB**(算法分,论文核心卖点)

---

## 1. 快速部署(Linux 云主机,2 × RTX 5090-32G;A800-80G 见 git log A800 分支)

> **5090 ×2 最小闭环**:BASE + SFT-ALG-25 + DPO-GvB。采样/训练全程 bf16(Blackwell + StarCoder2 都原生)。SFT 用 `ds_zero2_2gpu.json`(ZeRO-2 无 offload);DPO 默认 `ds_zero3_2gpu.json`(ZeRO-3 无 offload,policy+ref 各 6GB 必须切参);采样/评测用 `dp_sample.sh` / `dp_eval.sh`(数据并行起 2 个 vLLM 进程各占一卡,产物 cat 合并)。评分 `04_annotate --concurrency 50`。

> **路径约定**:仓库 clone 到 `/data/code/sketch-guided-clm-alignment`,产物全部落 `/data/work/out/`,模型在 `/data/models/`,APPS 数据在 `/data/datasets/APPS`,HF/MS cache 用 `/data/hf_cache`。根盘 `/` 只剩 ~24GB,放任何大件都会爆。

```bash
# --- 0. 准备目录(只跑一次) ---
mkdir -p /data/{code,models,datasets,hf_cache,work/out} /data/conda/envs

# --- 1. 克隆代码到 /data ---
cd /data/code
git clone -b feat/5090x2-adaptation https://github.com/qiaosiqi/sketch-guided-alignment.git sketch-guided-clm-alignment
cd sketch-guided-clm-alignment

# --- 2. Python 环境(conda 装到 /data,根盘装不下) ---
# 假设主机已有 miniconda/anaconda;若没有,先装到 /data/miniconda
conda env create -f environment-5090.yml -p /data/conda/envs/sketch5090
conda activate /data/conda/envs/sketch5090
# 验证 Blackwell 可见(应输出 (12, 0)):
python -c "import torch; print(torch.cuda.get_device_capability(0))"

# --- 3. 下载 APPS 数据集 (1.3GB,git 已排除) ---
cd /data/datasets
# 方式 A:Berkeley 官方
wget https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz
tar -xzf APPS.tar.gz   # 解出 APPS/raw/{train,test}/0000..4999/
# 方式 B:modelscope 镜像(国内更快)
# pip 已含 modelscope,可:
#   modelscope download --dataset codeparrot/apps --local_dir /data/datasets/APPS_ms
#   然后自己写一段把 ms dataset 转回 APPS 原始目录格式

# --- 4. 下载 base model(modelscope,国内云主机推荐) ---
modelscope download --model AI-ModelScope/starcoder2-3b --local_dir /data/models/StarCoder2-3B
# 若 modelscope 上没找到,改用 HF 镜像:
#   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download bigcode/starcoder2-3b \
#     --local-dir /data/models/StarCoder2-3B

# --- 5. 跑 smoke tests 验证环境 ---
cd /data/code/sketch-guided-clm-alignment
pytest v2/tests -v
# 应该 87 passed (Linux 上 9 个 runner test 也能跑)
```

**~/.bashrc 需要追加的环境变量**(永久化,登录即生效):

```bash
# Python / HuggingFace / Modelscope 缓存指向 /data,避免写满根盘
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export MODELSCOPE_CACHE=/data/hf_cache/modelscope

# GLM-4-Air judge API key(Phase 4 annotation 必需)
export GLM_API_KEY="your_zhipuai_api_key"

# 5090 ×2 PCIe 拓扑:NCCL 默认配置一般 OK;若训练 hang,取消下行注释
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

# 让 v2.* 模块路径可解析(也可在每次跑命令前 cd 到仓库根)
export PYTHONPATH=/data/code/sketch-guided-clm-alignment:${PYTHONPATH:-}

# 自动激活环境
source /data/conda/envs/sketch5090/bin/activate /data/conda/envs/sketch5090
```

---

## 2. 跑实验出数据(全流程,按顺序)

每一步都有断点续跑,中断后重跑命令会从上次产物继续。

> 所有命令的工作目录都假设是 `/data/code/sketch-guided-clm-alignment/`(`cd` 进去)。产物路径全部用 `/data/work/out/...` 绝对路径,不依赖 cwd。

### 2.1 数据预处理(秒级)

```bash
python -m v2.scripts.01_prepare_apps \
    --apps_root /data/datasets/APPS/raw \
    --out_dir /data/work/out/apps
# 产物:/data/work/out/apps/{train,val,test}.jsonl
# 过滤后只保留 interview 难度;train/val/test 实际题数以本步骤打印的统计为准
```

### 2.2 Pilot 验证(~20-40 分钟,双卡 vllm)

```bash
bash v2/scripts/dp_sample.sh /data/work/out/pilot \
    --problems_jsonl /data/work/out/apps/train.jsonl \
    --model_path /data/models/StarCoder2-3B \
    --n_problems 50 --n_per_temp 100 --temps 0.4 0.7 1.0

python -m v2.scripts.03_analyze_pilot --pilot_dir /data/work/out/pilot
```

`03_analyze_pilot` 会报 5 个指标 + 给出温度推荐。**根据结果决定 2.3 用单温度还是多温度。**

### 2.3 全量采样(双卡 vllm DP,~6-12 小时)

```bash
# train split:--do_timing 是 QvS 必需,主跑必加(代价 execution 阶段慢 3-10×)
bash v2/scripts/dp_sample.sh /data/work/out/sample_train \
    --problems_jsonl /data/work/out/apps/train.jsonl \
    --model_path /data/models/StarCoder2-3B \
    --n_problems 99999 --n_per_temp 100 \
    --temps 0.6 --do_timing   # 或多温度,按 pilot 结论改

# val split 同理
bash v2/scripts/dp_sample.sh /data/work/out/sample_val \
    --problems_jsonl /data/work/out/apps/val.jsonl \
    --model_path /data/models/StarCoder2-3B \
    --n_problems 99999 --n_per_temp 100 --temps 0.6 --do_timing
```

> `dp_sample.sh` 会起 2 个 vLLM 进程各 pin 一张卡,题目对半切片,跑完 cat 合并产物到 OUT_DIR 下。中断重跑安全(每个 shard 子目录 append 写入)。

### 2.4 GLM-4-Air 评分(API,~10-30 小时,取决于并发和阈值)

**关键参数 `--pass_threshold`**:只评 pass_ratio ≥ 此值的解,省 API 钱。
- `0.0`:全评(最贵,得到完整分布)
- `0.5`:推荐,与 `θ_pass_gvb=0.5` 严格对齐 —— 所有可能进 GvB 的解恰好都被评分,且不会多评一份(低于 0.5 的解 GvB 用不到)

```bash
python -m v2.scripts.04_annotate \
    --problems_jsonl /data/work/out/apps/train.jsonl \
    --sample_dir /data/work/out/sample_train \
    --pass_threshold 0.5 --alpha 0.4 --concurrency 50

python -m v2.scripts.04_annotate \
    --problems_jsonl /data/work/out/apps/val.jsonl \
    --sample_dir /data/work/out/sample_val \
    --pass_threshold 0.5 --alpha 0.4 --concurrency 50
```

API 调用粗算:`训练题数 × 100 解 × 2 次调用`(`pass_threshold=0`,全评)。设 `0.5` 后按解的通过率分布大致砍到 1/3~1/5。中断重跑会续。

### 2.5 合并训练数据(分钟级)

```bash
python -m v2.scripts.05_merge \
    --problems_jsonl /data/work/out/apps/train.jsonl \
    --sample_dir /data/work/out/sample_train \
    --out /data/work/out/datasets/train/merged.jsonl

python -m v2.scripts.05_merge \
    --problems_jsonl /data/work/out/apps/val.jsonl \
    --sample_dir /data/work/out/sample_val \
    --out /data/work/out/datasets/val/merged.jsonl
```

### 2.6 SFT 训练(~4-8 小时,DeepSpeed ZeRO-2 双卡 5090)

```bash
# 论文里的 ALG 风格(按算法分排 top-25%)
deepspeed --num_gpus 2 -m v2.scripts.06_train_sft \
    --train_merged /data/work/out/datasets/train/merged.jsonl \
    --val_merged /data/work/out/datasets/val/merged.jsonl \
    --model_path /data/models/StarCoder2-3B \
    --output_dir /data/work/out/runs/sft_alg_top25 \
    --sort_by algo_final --top_p 25 --augment True \
    --num_train_epochs 10 \
    --ds_config v2/configs/ds_zero2_2gpu.json   # ZeRO-2 无 offload + bf16

# 对照实验:SPD(按 runtime),PASS(按 pass_ratio)
# 只改 --sort_by 即可
```

### 2.7 DPO 训练(~4-8 小时,接 SFT 检查点继续)

```bash
# 主实验:GvB(本工作核心);DPO 走 ZeRO-3(policy+ref 各 6GB 必须切参才能装下 32GB)
deepspeed --num_gpus 2 -m v2.scripts.07_train_dpo \
    --train_merged /data/work/out/datasets/train/merged.jsonl \
    --val_merged /data/work/out/datasets/val/merged.jsonl \
    --model_path /data/work/out/runs/sft_alg_top25 \
    --output_dir /data/work/out/runs/dpo_gvb \
    --task gvb --augment True \
    --ds_config v2/configs/ds_zero3_2gpu.json   # ZeRO-3 无 offload + bf16

# 对照:PvF / QvS / ALL —— 只改 --task 和 --output_dir
# OOM 时加 --use_lora(也可切回 ds_zero2_2gpu.json,LoRA 模式下 ref 与 policy 共享)
```

### 2.8 评测(test 集,~3-6 小时,双卡 vllm DP)

```bash
bash v2/scripts/dp_eval.sh /data/work/out/evals/dpo_gvb \
    --problems_jsonl /data/work/out/apps/test.jsonl \
    --model_path /data/work/out/runs/dpo_gvb \
    --n_per_temp 100 --temps 0.6 --do_timing

# 同样跑 baseline / SFT-only / 其他 DPO 任务做对比
bash v2/scripts/dp_eval.sh /data/work/out/evals/base \
    --problems_jsonl /data/work/out/apps/test.jsonl \
    --model_path /data/models/StarCoder2-3B \
    --n_per_temp 100 --temps 0.6 --do_timing
bash v2/scripts/dp_eval.sh /data/work/out/evals/sft_alg \
    --problems_jsonl /data/work/out/apps/test.jsonl \
    --model_path /data/work/out/runs/sft_alg_top25 \
    --n_per_temp 100 --temps 0.6 --do_timing
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
| DPO-PvF | `--task pvf`(二元正确性主信号:1.0 vs 0.0,从 SFT-ALG-25 继续) |
| DPO-QvS | `--task qvs`(runtime,仅 pass_ratio=1.0 内比) |
| DPO-GvB | `--task gvb`(算法分,**论文核心卖点**) |
| DPO-ALL | `--task all`(pvf / qvs / gvb 随机回退,多目标融合) |

---

## 4. 框架状态

- [x] Phase 1 — Data schema / prompts / APPS loader
- [x] Phase 2 — Execution (stdio + fncall,partial credit,稳定测时)
- [x] Phase 3 — Sampling(两段式,vllm + HF 双后端)
- [x] Phase 4 — Judge(GLM-4-Air + 9 维 rubric)+ Merge
- [x] Phase 5a — SFT(trl 0.11+,DynamicSFTCollator)
- [x] Phase 5b — DPO(PvF / QvS / GvB / ALL,动态 pair 采样)
- [x] Phase 5c — Evaluation(pass@k + mean_pass_ratio + runtime + algo)
- [x] Smoke tests — 87 个用例(78 Win + 9 Linux-only)

各 phase 的设计细节见 [`v2/README.md`](v2/README.md);测试细节见 [`v2/tests/README.md`](v2/tests/README.md);上层代码导览见 [`CLAUDE.md`](CLAUDE.md)。

---

## 5. 排错速查

| 症状 | 原因 / 解决 |
|---|---|
| `RuntimeError: no kernel image is available` 或 `CUDA capability sm_120 is not compatible` | 装错了 PyTorch wheel(默认 PyPI 是 cu126,Blackwell 需要 cu128)。重装 `pip install torch==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128`。`environment-5090.yml` 已指定。 |
| vLLM 启动卡死 / 段错误(5090) | 升级 vllm 至 0.8.5+。若仍异常,设 `VLLM_USE_V1=0` 回退老 engine,或临时改用 `--prefer_backend hf`。 |
| `bf16 not supported` | 不应在 5090 上发生。若发生,确认实际跑的是 5090 卡(`nvidia-smi`)而非主机其他低端卡。 |
| NCCL hang / `unhandled cuda error` 在 2 卡 ZeRO-2/3 中 | 5090 走 PCIe 时 P2P 可能不稳。先尝试 `export NCCL_P2P_DISABLE=1`,再不行加 `NCCL_SHM_DISABLE=1`;严重时设 `NCCL_DEBUG=INFO` 看具体阶段。 |
| DPO OOM | 默认 ZeRO-3 装不下时:① 改 `--per_device_train_batch_size 1 --gradient_accumulation_steps 8`(有效 batch 仍 = 16);② 加 `--use_lora`;③ 极端情况切 `ds_zero2_2gpu.json` + `--use_lora`(ref 与 policy 共享,显存最省)。 |
| 根盘 `/` 写满 | 必然是 HF cache 或 checkpoint 落 `/` 了。检查 `HF_HOME=/data/hf_cache`、`--output_dir` 是否绝对路径到 `/data`。 |
| 评分太慢 | 确认 `--pass_threshold 0.5`(别用 0.0 全评)、`--concurrency 50`。 |
| `signal.SIGALRM` AttributeError | 在 Windows 上跑 execution → 必须用 Linux |
| 模型 import trust_remote_code 警告 | StarCoder2 需要 `trust_remote_code=True`,代码已设 |

---

## License

本项目部分代码基于 Code-Optimise (Apache 2.0),保留相应版权声明。
