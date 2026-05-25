#!/usr/bin/env bash
# 4090 双卡云容器主机的环境变量。每开新 tmux session,先 `source` 这份。
#
# 配套硬件:RTX4090-PCIe5.0 × 2 (24G/卡, sm_89),28 vCPU,内存 240GB。
# 系统盘 /(overlay)只有 30G,绝不能让权重/cache/conda 落到根盘 —— 全部走
# JuiceFS 数据盘 /root/shared-nvme/(500G)。
#
# 用法:
#   source v2/scripts/env_4090.sh
#   conda activate /root/shared-nvme/conda/envs/sketch4090

# ---- 清掉 NGC 容器在 base shell 里硬钉的 PIP_CONSTRAINT ----
# NGC 镜像 base 环境写死 PIP_CONSTRAINT=/etc/pip/constraint.txt,把 packaging/setuptools/
# urllib3/idna/… 等传递依赖钉到 NVIDIA NGC 的老版本。conda activate 不会主动清它,
# 任何忘加前缀的 pip install 都会把 setup_env.sh 协商出来的新版本降级回去
# (实际遇过:packaging 26→23.2 触发 ray/wheel 版本冲突)。每次 source 此文件无脑清掉。
unset PIP_CONSTRAINT

# ---- 实验产物根目录(对应 RUN_PLAN.md 里的 $WORK 等) ----
export WORK=/root/shared-nvme/work/out
export APPS=$WORK/apps
export DATASETS=$WORK/datasets
export RUNS=$WORK/runs
export EVALS=$WORK/evals
export MODEL=/root/shared-nvme/models/StarCoder2-3B

# ---- 各类 cache:必须导出到 NVMe,不能落系统盘 ----
# dp_sample.sh / dp_eval.sh 内部按 :- 取这三个变量作 BASE,再拼 shard0/shard1 子目录。
export TORCHINDUCTOR_CACHE_DIR=/root/shared-nvme/cache/torchinductor
export TRITON_CACHE_DIR=/root/shared-nvme/cache/triton
export VLLM_CACHE_ROOT=/root/shared-nvme/cache/vllm
# HuggingFace / transformers 模型下载落点(modelscope 走 cache_dir 不读这个,但 from_pretrained 读)
export HF_HOME=/root/shared-nvme/cache/hf
export TRANSFORMERS_CACHE=/root/shared-nvme/cache/hf

# ---- 训练 / 采样运行时 ----
# Phase 6 剩余 eval 阶段已不再调 GLM-4-Air,不需要 GLM_API_KEY;若要补标注再设。
# export GLM_API_KEY=...

# 5090/4090 都需要:expandable_segments 缓解显存碎片(DPO precompute_ref 后释放 ref 时尤其有效)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DeepSpeed 启动时跳过 nvcc 检测(我们用 prebuilt wheel,机内无 cuda toolkit)
export DS_SKIP_CUDA_CHECK=1
# 训练里 tokenizer 已 import 在 fork 之前完成,显式关掉它的多线程警告
export TOKENIZERS_PARALLELISM=false

echo "[env_4090] WORK=$WORK"
echo "[env_4090] cache → /root/shared-nvme/cache/{torchinductor,triton,vllm,hf}"
echo "[env_4090] 别忘了: conda activate /root/shared-nvme/conda/envs/sketch4090"
