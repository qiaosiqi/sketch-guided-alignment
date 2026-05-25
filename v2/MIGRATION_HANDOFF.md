# Migration Handoff: 5090×2 → 4090(2026-05-25)

科研经费调整,从 5090×2(32G/卡)迁到 4090(具体配置另开会话定)。本文件冻结当前工作状态、列出**必须从 5090 服务器下载**到本地的产物,以及 4090 上线后的恢复步骤。

> 配套文件: `RUN_PLAN.md`(总活文档)、`BUGS.md`(所有已知 bug)。代码 / 配置 / 文档已在 git(branch `feat/5090x2-adaptation`,HEAD `274a5dc`),迁移期间不动代码。

---

## 1. 冻结时刻的工作状态

| 阶段 | 状态 | 备注 |
|---|---|---|
| Phase 0 BASE eval | ✅ | pass@1=0.0108 / pass@10=0.0541 |
| Phase 1-2 采样+标注 (train+val) | ✅ | GLM-4-Air 评分已花费,**不可重做** |
| Phase 3 merge | ✅ | train 760 题 / val 84 题 |
| Phase 4 SFT-ALG-25 | ✅ | best=ckpt-240,eval_loss=0.3992 |
| Phase 5a-d DPO ×4 | ✅ | gvb/pvf/qvs/all 全部完成,best ckpt 已确定 |
| Phase 6 eval sft | ✅ | pass@1=0.0127 / pass@10=0.0530 / pass@100=0.1251(26.5h on 5090×2) |
| Phase 6 eval dpo_gvb | ⏳ 运行中(冻结时启动 ~3h) | 见 §3 决策 |
| Phase 6 eval dpo_pvf/qvs/all | ❌ 未跑 | 留到 4090 |

---

## 2. 必须从 5090 下载的产物(优先级排序)

服务器路径基于 `/data/work/out/`。先 `ssh` 到服务器,用 `rsync` 拉到本地。**Tier-1 是不可重做的,必须先拉**。

### Tier 1 —— 不可重做(GLM API 钱 / 训练时间)

| 路径 | 体积 | 重做代价 |
|---|---|---|
| `/data/work/out/datasets/train/merged.jsonl` | ~100MB | GLM-4-Air ×760 题 ×N 候选 ≈ $$ |
| `/data/work/out/datasets/val/merged.jsonl` | ~10MB | GLM-4-Air ×84 题 |
| `/data/work/out/runs/sft_alg_top25/best/` | ~5.8GB | 5090×2 86min |
| `/data/work/out/runs/dpo_gvb/best/` | ~5.8GB | 5090×2 29min |
| `/data/work/out/runs/dpo_pvf/best/` | ~5.8GB | 5090×2 ~30min |
| `/data/work/out/runs/dpo_qvs/best/` | ~5.8GB | 5090×2 ~50min |
| `/data/work/out/runs/dpo_all/best/` | ~5.8GB | 5090×2 ~30min |
| `/data/work/out/runs/*/trainer_state.json` | <1MB ×5 | 训练曲线,论文 figure 用 |
| `/data/work/out/apps/{train,val,test}.jsonl` | ~50MB | 确定性可重做,但顺手带上 |
| `/data/work/out/evals/sft_alg_top25/metrics.json` | <1KB | 已 sanity 过的 SFT 数字 |

**Tier-1 合计 ≈ 35GB。**

### Tier 2 —— 重做可行但费时(分析 / 论文 figure 备份)

| 路径 | 体积 | 用途 |
|---|---|---|
| `/data/work/out/evals/sft_alg_top25/{sketches,codes,exec}.jsonl` | ~1.3GB | 论文里如做 case study / 误差分析需要原始生成 |
| `/data/work/out/main_train/{codes,exec,scores}.jsonl` | ~3-5GB | 重算 merged.jsonl 的备份(避免 merged 丢失) |
| `/data/work/out/main_val/{codes,exec,scores}.jsonl` | ~1GB | 同上 |
| `/data/work/out/evals/base/{metrics,sketches,codes,exec}.jsonl` | ~1.5GB | BASE 数字回填论文 |

**Tier-2 合计 ≈ 8GB。**(建议一并拉,反正才 8G)

### 不要下载

- `/data/work/out/runs/*/checkpoint-*/`(中间 ckpt;只要 `best/` 软链所指的那个)
  - 注:`best/` 是相对路径 symlink 指向 `checkpoint-N`,**下载时务必用 `rsync -L` 把 symlink 解引用为真正的 ckpt 目录**,否则到 4090 上软链断掉。
- `/data/cache/{vllm,torchinductor,triton}/`(全部 cache,4090 上要重新编)
- `~/.cache/{vllm,huggingface}/`(同上)
- 任何 `_shard0` / `_shard1` 子目录(已合并到父目录的最终 jsonl)

---

## 3. dpo_gvb eval 是否在 5090 跑完?

**决策点**:dpo_gvb eval 冻结时启动 ~3h,按 SFT eval 经验还需要 ~25h。

| 方案 | 适合场景 | 代价 |
|---|---|---|
| **A. 让它在 5090 跑完再迁** | 5090 还能用 ≥ 28h | 拿到 headline 数字,迁移更从容 |
| **B. 立刻 Ctrl-C,4090 上重跑** | 5090 ASAP 收回 | 4090 上多花 ~一天(且 n_per_temp 可降到 30 缩到 ~8h) |
| **C. 让它后台跑,同时本地开始 Tier-1 下载** | 不确定 5090 还能用多久 | 推荐 —— 不浪费现成的 28h 进度,下载和 eval 并行 |

**推荐 C**。下载 35GB 在带宽 50Mbps 下约 1.5h,远快于 eval 完成,完全 overlap。

如果选 A 或 C,在 dpo_gvb 跑完后**追加下载** `/data/work/out/evals/dpo_gvb/`(metrics + 三个 jsonl ≈ 1.4GB),然后才关 5090。

---

## 4. 下载命令(在本地执行)

约定本地落点 `D:/CODE/myClaude/sketch-guided-clm-alignment/handoff_artifacts/`(已加入 `.gitignore`,不会污染 repo)。

```powershell
# Windows PowerShell —— 走 WSL 的 rsync 最稳;不行就用 scp -r
$LOCAL = "D:\CODE\myClaude\sketch-guided-clm-alignment\handoff_artifacts"
New-Item -ItemType Directory -Force $LOCAL | Out-Null

# 假设 ssh alias 是 gpu5090,如果不是改成 user@host
# -L 把 symlink 解引用(best/ → 实际 checkpoint 目录)
# -avzh 压缩 + 显示;--progress 看进度;--partial 断点续传
ssh gpu5090 'tar -czf - -C /data/work/out \
    datasets \
    apps \
    runs/sft_alg_top25/best \
    runs/sft_alg_top25/trainer_state.json \
    runs/dpo_gvb/best runs/dpo_gvb/trainer_state.json \
    runs/dpo_pvf/best runs/dpo_pvf/trainer_state.json \
    runs/dpo_qvs/best runs/dpo_qvs/trainer_state.json \
    runs/dpo_all/best runs/dpo_all/trainer_state.json \
    evals/sft_alg_top25/metrics.json \
    evals/base \
    --dereference' | tar -xzf - -C $LOCAL
```

或者更稳的分步 rsync(走 WSL):

```bash
# 在 WSL / Git Bash 里
LOCAL=/mnt/d/CODE/myClaude/sketch-guided-clm-alignment/handoff_artifacts
mkdir -p $LOCAL

# Tier 1
rsync -avzhL --progress --partial \
    gpu5090:/data/work/out/datasets/ \
    $LOCAL/datasets/

rsync -avzhL --progress --partial \
    gpu5090:/data/work/out/apps/ \
    $LOCAL/apps/

for run in sft_alg_top25 dpo_gvb dpo_pvf dpo_qvs dpo_all; do
    rsync -avzhL --progress --partial \
        gpu5090:/data/work/out/runs/$run/best/ \
        $LOCAL/runs/$run/best/
    rsync -avzh --progress \
        gpu5090:/data/work/out/runs/$run/trainer_state.json \
        $LOCAL/runs/$run/
done

rsync -avzh --progress \
    gpu5090:/data/work/out/evals/sft_alg_top25/metrics.json \
    $LOCAL/evals/sft_alg_top25/

# Tier 2(可选)
rsync -avzhL --progress --partial \
    gpu5090:/data/work/out/evals/base/ \
    $LOCAL/evals/base/

rsync -avzh --progress \
    "gpu5090:/data/work/out/evals/sft_alg_top25/{sketches,codes,exec}.jsonl" \
    $LOCAL/evals/sft_alg_top25/

rsync -avzh --progress \
    "gpu5090:/data/work/out/main_train/{codes,exec,scores}.jsonl" \
    $LOCAL/main_train/

rsync -avzh --progress \
    "gpu5090:/data/work/out/main_val/{codes,exec,scores}.jsonl" \
    $LOCAL/main_val/
```

下载完核验:

```powershell
# 看体积
Get-ChildItem -Recurse $LOCAL | Measure-Object -Property Length -Sum

# 抽查模型文件
ls "$LOCAL\runs\sft_alg_top25\best\"   # 应含 model.safetensors / config.json / tokenizer*
ls "$LOCAL\runs\dpo_gvb\best\"

# 抽查 merged 数据
(Get-Content "$LOCAL\datasets\train\merged.jsonl" -Head 1) | ConvertFrom-Json | Format-List
```

---

## 5. 4090 上线后的恢复 checklist

(具体硬件 / 软件配置在另一个会话里和你对齐后,在本节填实)

### 5.1 已知 4090 vs 5090 关键差异(待和你的具体配置对齐)

| 维度 | 5090 | 4090 | 影响 |
|---|---|---|---|
| 架构 | Blackwell sm_120 | Ada Lovelace sm_89 | 4090 是成熟架构,torch / vllm 预编译 wheel 都直接有;**不必再钉 torch 2.7+cu128 / vllm 0.8.5**,可回到更稳定的 torch 2.4-2.5 + cu121 + vllm 0.6-0.10 |
| VRAM/卡 | 32GB | **24GB** | DPO 之前用 ZeRO-2 + offload + precompute_ref_log_probs 才装下;4090 上只会更紧。若卡数 ≤ 2,可能要强制 ZeRO-3 或加大 grad-accum |
| bf16 原生 | ✓ | ✓ | 训练精度不变 |
| 卡数 | 2 | 待定 | 决定 launcher (dp_sample.sh / dp_eval.sh 的 shard 数) |

### 5.2 上线步骤(模板)

1. 装环境:`bash v2/scripts/setup_env.sh`(可能要把 torch / vllm 版本号改回 cu121 稳定线)
2. 拉代码:`git clone ...` + `git checkout feat/5090x2-adaptation`
3. 把 `handoff_artifacts/` 上传到 4090 的 `$WORK = /xxx/work/out` 对应目录
4. **重建 best 软链**(因为 rsync -L 把它解引用了,变成真目录,这步省了)—— 直接 `--model_path .../best` 就能用
5. 重设环境变量(`RUN_PLAN.md` §通用环境)
6. 跑一次最小 smoke test:
   ```bash
   # 5 个 test 题 + 1 sample,验证模型能加载、能采、能 exec
   bash v2/scripts/dp_eval.sh /tmp/smoke \
       --problems_jsonl $APPS/test.jsonl \
       --model_path $RUNS/sft_alg_top25/best \
       --n_per_temp 1 --temps 0.6 --n_problems 5
   ```
   通过 → 进 Phase 6 剩余 eval

### 5.3 Phase 6 剩余 eval 计划

- **方案推荐**:n_per_temp 从 100 降到 **30**(已在 RUN_PLAN 里讨论),pass@1/pass@10 仍可估,放弃 pass@100。单 eval 从 ~26h 降到 ~8h,4 个 eval ~32h
- 命令模板:
  ```bash
  for run in dpo_gvb dpo_pvf dpo_qvs dpo_all; do
      bash v2/scripts/dp_eval.sh $EVALS/$run \
          --problems_jsonl $APPS/test.jsonl \
          --model_path $RUNS/$run/best \
          --n_per_temp 30 --temps 0.6
  done
  ```
  (如果选了上面方案 C 已经跑完 dpo_gvb,这里去掉它)

---

## 6. 不需要下载 / 不需要带走的东西

- 整个 `/data/code/sketch-guided-clm-alignment` 代码目录 —— git 已同步
- 所有 cache 目录(`/data/cache/*`、`~/.cache/*`)
- 所有 `checkpoint-N`(只要 `best/`)
- conda env(`sketch5090`)—— 4090 上 setup_env.sh 重建

## 7. 关掉 5090 前的最终 checklist

- [ ] Tier-1 全部下载完成,体积 ≈ 35GB
- [ ] (可选)Tier-2 下载完成
- [ ] dpo_gvb eval 已完成 → 追加下载 `evals/dpo_gvb/` ;或决定放弃
- [ ] 本地 `handoff_artifacts/` 抽查:5 个 `best/` 目录都含 `model.safetensors`、merged.jsonl 行数对得上
- [ ] 本文件 + RUN_PLAN.md + BUGS.md 已 `git commit && git push`
- [ ] 5090 上的活跃 tmux session 状态确认无未保存输出(`trainer_state.json` 已落盘 / eval 跑完)
