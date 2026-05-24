# 主跑运行计划(活文档)

主实验从 BASE 评测起,到 5 模型对比收尾。每个阶段跑完更新本文件的状态表 + 结果日志,核查点打勾后才进下一阶段。

**研究目标锚点**:验证 sketch-guided 两阶段生成 + GvB(algo-quality)偏好对比 PvF(binary 正确性)在 pass@k / runtime / algo-score 上的增益。

---

## 状态表

| Phase | 任务 | 状态 | 关键产物路径 |
|-------|------|------|--------------|
| 0 | BASE 评测(锚点) | ☑ 完成(2026-05-23) | `$EVALS/base/metrics.json` |
| 1a | 主采样 train | ☑ 完成(2026-05-23,761 题×50 样本,98.6% runtime 覆盖) | `$WORK/main_train/` |
| 1b | 主采样 val | ☑ 完成(2026-05-23,84 题×50 样本,98.2% runtime 覆盖) | `$WORK/main_val/` |
| 2a | GLM-4-Air 标注 train | ☑ 完成(2026-05-23,99.99% parsable,pass-fail diff=1.83) | `$WORK/main_train/scores.jsonl` |
| 2b | GLM-4-Air 标注 val | ☑ 完成(2026-05-23,100% parsable,pass-fail diff=1.44) | `$WORK/main_val/scores.jsonl` |
| 3 | 合并 train + val | ☑ 完成(2026-05-23,train 760 题,val 84 题) | `$DATASETS/{train,val}/merged.jsonl` |
| 4 | SFT-ALG-25 训练 | ☑ 完成(2026-05-23,240 steps/86min,best=checkpoint-240,eval_loss=0.3992,acc=86.97%) | `$RUNS/sft_alg_top25/best` |
| 5a | DPO-GvB(headline) | ☑ 完成(2026-05-24,236 steps/29min,eval_loss=0.520,eval_margins=+0.60,grad_norm末段51✓) | `$RUNS/dpo_gvb/best` |
| 5b | DPO-PvF(baseline) | ☑ 完成(2026-05-24,best=ep1=ckpt-166,eval_loss=0.704,margins=+0.029;ep2+ trajectory 丢失) | `$RUNS/dpo_pvf/best` |
| 5c | DPO-QvS(timing) | ☑ 完成(2026-05-24,best=ep3=ckpt-345,eval_loss=0.615,margins=+0.38,U 形 ep5 反弹) | `$RUNS/dpo_qvs/best` |
| 5d | DPO-ALL(fallback) | ☑ 完成(2026-05-24,best=ep1=ckpt-167,eval_loss=0.71→1.23 恶化,负面对照) | `$RUNS/dpo_all/best` |
| 6 | 5 模型评测 | ☐ 未跑 | `$EVALS/{sft_alg_top25,dpo_*}/metrics.json` |

**约定**:勾 ☐→☑ 表示主进程完成;勾选人工核查点 (✋) 后才允许进下一阶段。

---

## 操作规程:tmux(强烈推荐)

主跑全程多任务都 ≥ 1 小时,本地电脑可能关掉,**所有命令都在 tmux 里跑**。tmux session 独立于 SSH 连接,关电脑/断网/重连都不影响。

```bash
# 一次性安装
which tmux || sudo apt install -y tmux

# 启动 / 接入 main session
tmux new -s main       # 第一次
tmux attach -t main    # 重连
tmux ls                # 列已有 session

# 在 tmux 里 export + 跑命令,前台直接跑(不用 nohup)

# 暂离不杀进程:Ctrl-B  然后 D
# 滚动看历史:  Ctrl-B  然后 [   (q 退出)
# 分屏(可选): Ctrl-B  然后 "   水平分;Ctrl-B 然后 % 竖分
# 切 pane:     Ctrl-B  然后 方向键
```

**日志查看渠道**:
| 来源 | 用法 |
|------|------|
| tmux 实时输出 | `tmux attach -t main` |
| shard 落盘日志(采样/评测) | `tail -f $WORK/.../_shard0/stderr.log` |
| 训练 trainer_state | `cat $RUNS/<run>/trainer_state.json \| python -m json.tool` |
| 行数探活 | `wc -l $WORK/.../codes.jsonl` |
| GPU 占用 | `nvidia-smi -l 2` |

## 通用环境

新会话开头先 export 一次(每个 Phase 开跑前确认 `echo $WORK` 非空):

```bash
cd /data/code/sketch-guided-clm-alignment
conda activate sketch5090

export WORK=/data/work/out
export APPS=$WORK/apps
export DATASETS=$WORK/datasets
export RUNS=$WORK/runs
export EVALS=$WORK/evals
export MODEL=/data/models/StarCoder2-3B    # 按实际权重路径调整

export GLM_API_KEY=...                      # Phase 2 之前必填
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1
export TOKENIZERS_PARALLELISM=false
```

**超参策略**:研究语义相关的训练超参全部用脚本默认值
(epoch=10, LR=5e-7, beta=0.1, tau=6.0, theta_pass_gvb=0.5, alpha=0.4,
sort_by=algo_final, top_p=25, augment=True)。
仅 batch_size / gradient_accumulation 按 5090 32G 适配
(SFT: per_dev=1×accum=16;DPO: per_dev=1×accum=8),这不影响研究结论。

---

## Phase 0:BASE 评测(锚点)

**目的**:在训练任何模型前,先把 BASE StarCoder2-3B 在 test 集上的 pass@k / runtime 定下来,后续所有训完的模型都和它对比。

**先单独跑这一步**,等结果回到主对话 sanity check 后再进 Phase 1。

```bash
bash v2/scripts/dp_eval.sh $EVALS/base \
    --problems_jsonl $APPS/test.jsonl \
    --model_path $MODEL \
    --n_per_temp 100 --temps 0.6 \
    --do_timing
```

预计 ~1-2h(test 集 ~500-700 题 × 100 样本 × 双卡)。

### ✋ 核查点 0

```bash
cat $EVALS/base/metrics.json | python -m json.tool
ls -la $EVALS/base/
wc -l $EVALS/base/exec.jsonl
tail -20 $EVALS/base/_shard0/stderr.log
```

通过条件:
- [ ] `metrics.json` 含 `pass@1`、`pass@10`、`mean_runtime_ns`
- [ ] `pass@1 ∈ [0.05, 0.20]`(StarCoder2-3B × APPS interview 合理区间)
- [ ] shard 日志无 OOM / vLLM fallback
- [ ] `exec.jsonl` 行数 ≈ test 题数 × 100(单 temp)

### 结果日志

```
BASE pass@1           = 0.0108   (1.08%)
BASE pass@10          = 0.0541   (5.41%)
BASE mean_pass_ratio  = 0.1020   (10.20%)
BASE n_problems       = 2998
BASE samples/problem  = 99.67
```

mean_runtime_ns 字段忽略 —— 本次 eval 不带 --do_timing,字段中是早期 --do_timing
试跑残留的几条旧 exec 数据,无意义。论文只在训练后模型间比较 runtime。

执行耗时:~35h(BUG-001 + BUG-002 修复后实测)。

最终矩阵将填入 Phase 6。

---

## Phase 1:主采样 train + val(两步串行)

**目的**:用 BASE 模型在 train + val 上生成 (sketch, code) 候选池,带 timing(QvS 必须)。

**规模决定(2026-05-23)**:BASE eval 实测 ~35h(3000 题 × 100 样本无 timing)
让人意识到 Phase 1 全量 + timing 会跑 4-5 天,风险过大。采用"减半 × 减半"方案:
**`n_per_temp=50`,题数减半**,预计 Phase 1 ~30-40h。trade-off:全 pass 解可能变少
(尤其 QvS 池子变薄),但 GvB / PvF 训练量仍充足。

### 1a) train 采样

```bash
# 先算半题数
HALF_TRAIN=$(( $(wc -l < $APPS/train.jsonl) / 2 ))
echo "train half = $HALF_TRAIN"

bash v2/scripts/dp_sample.sh $WORK/main_train \
    --problems_jsonl $APPS/train.jsonl \
    --model_path $MODEL \
    --n_problems $HALF_TRAIN --n_per_temp 50 --temps 0.6 \
    --do_timing
```

在 tmux 里前台直接跑;离开按 `Ctrl-B D`。监控另开 pane:`tail -f $WORK/main_train/_shard0/stderr.log`。预计 ~30-40h。

### 1b) val 采样

```bash
HALF_VAL=$(( $(wc -l < $APPS/val.jsonl) / 2 ))
echo "val half = $HALF_VAL"

bash v2/scripts/dp_sample.sh $WORK/main_val \
    --problems_jsonl $APPS/val.jsonl \
    --model_path $MODEL \
    --n_problems $HALF_VAL --n_per_temp 50 --temps 0.6 \
    --do_timing
```

预计 ~3-5h。

### ✋ 核查点 1

```bash
for d in main_train main_val; do
    echo "=== $d ==="
    wc -l $WORK/$d/*.jsonl
done
head -1 $WORK/main_train/codes.jsonl | python -m json.tool | head -20
head -1 $WORK/main_train/exec.jsonl  | python -m json.tool | head -20
```

通过条件:
- [ ] `chosen_problems.jsonl` 行数 ≈ train.jsonl / val.jsonl 行数
- [ ] `sketches.jsonl` 行数 ≈ problems × 100
- [ ] `codes.jsonl` 行数 ≈ `sketches.jsonl`(两阶段每 sketch 一 code)
- [ ] `exec.jsonl` 行数 ≈ `codes.jsonl`
- [ ] 抽样的 code 含完整 Python,exec 含 `pass_ratio`,全 pass 解含 `runtime_ns`

---

## Phase 2:GLM-4-Air 9D 标注

**目的**:对每个 (sketch, code) 对打 4 sketch + 5 code = 9 维分,合成 `algo_final`(GvB 训练信号)。

```bash
# 确认 key
[ -n "$GLM_API_KEY" ] || { echo "GLM_API_KEY 未设置"; exit 1; }

python -m v2.scripts.04_annotate \
    --problems_jsonl $APPS/train.jsonl \
    --sample_dir $WORK/main_train \
    --concurrency 50

python -m v2.scripts.04_annotate \
    --problems_jsonl $APPS/val.jsonl \
    --sample_dir $WORK/main_val \
    --concurrency 50
```

预计 ~1-3h(GLM-4-Air 限速决定)。遇 429 调低 `--concurrency`。

### ✋ 核查点 2

```bash
for d in main_train main_val; do
    echo "=== $d ==="
    echo "codes:  $(wc -l < $WORK/$d/codes.jsonl)"
    echo "scores: $(wc -l < $WORK/$d/scores.jsonl)"
done
head -1 $WORK/main_train/scores.jsonl | python -m json.tool
```

通过条件:
- [ ] `scores` 行数 == `codes` 行数(没漏标)
- [ ] `algo_final ∈ [0, 10]`,S1-S4 / C1-C5 都是 0-10 整数
- [ ] 抽 3-5 条高分 sketch 人工感官检查,GLM 评分大致合理

---

## Phase 3:合并

```bash
mkdir -p $DATASETS/train $DATASETS/val

python -m v2.scripts.05_merge \
    --problems_jsonl $APPS/train.jsonl \
    --sample_dir $WORK/main_train \
    --out $DATASETS/train/merged.jsonl

python -m v2.scripts.05_merge \
    --problems_jsonl $APPS/val.jsonl \
    --sample_dir $WORK/main_val \
    --out $DATASETS/val/merged.jsonl
```

### ✋ 核查点 3

```bash
wc -l $DATASETS/train/merged.jsonl $DATASETS/val/merged.jsonl
python -c "
import json
for split in ['train', 'val']:
    with open(f'$DATASETS/{split}/merged.jsonl') as f:
        p = json.loads(next(f))
    print(f'[{split}] answers/problem={len(p[\"answers\"])} '
          f'keys={list(p[\"answers\"][0].keys())} '
          f'first scores={p[\"answers\"][0].get(\"scores\")}')"
```

通过条件:
- [ ] 行数:train ~problems 数(已剔除零 valid),val 类似
- [ ] 每题 `answers` 非空(merge 已丢空)
- [ ] `answers[0]` 含 `pass_ratio`、`runtime_ns`(全 pass 时)、`scores.algo_final`

---

## Phase 4:SFT-ALG-25 训练

**单独 1 个 SFT 模型**,作为后续 DPO 的接力起点。

```bash
deepspeed --num_gpus 2 --module v2.scripts.06_train_sft \
    --train_merged $DATASETS/train/merged.jsonl \
    --val_merged $DATASETS/val/merged.jsonl \
    --model_path $MODEL \
    --output_dir $RUNS/sft_alg_top25 \
    --sort_by algo_final --top_p 25 --augment True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --ds_config v2/configs/ds_zero3_2gpu.json
```

⚠️ `--per_device_eval_batch_size 1` 是 5090 32G 必须显式覆盖的 —— 脚本默认是 4,
但 StarCoder2-3B logits 在 cross_entropy 里要 fp32 中间张量
(`4 × 2048 × 49k × 4B ≈ 1.6 GiB`),eval batch=4 会 OOM。

预计 ~6-10h。

### ✋ 核查点 4

```bash
ls -la $RUNS/sft_alg_top25/best
python -c "
import json
s = json.load(open('$RUNS/sft_alg_top25/trainer_state.json'))
print('best:', s.get('best_model_checkpoint'), 'metric:', s.get('best_metric'))
print('--- last 5 log entries ---')
for r in s['log_history'][-5:]: print(r)
"
```

通过条件:
- [ ] `best/` symlink 指向某 `checkpoint-N`
- [ ] `eval_loss` 走势单调下降或中段 plateau,无后段大幅反弹
- [ ] 最终 `eval_loss < initial`

### 结果日志

```
SFT best ckpt    = checkpoint-240
SFT best eval_loss = 0.3992
SFT token acc    = 86.97%
SFT wall-clock   = 86min  (240 steps,  per_dev=1×accum=16×2gpu,augment=True)
```

注:240 steps = 760 题 / 32 batch ≈ 24 步/epoch × 10 epoch。SFT 的
`--augment True` 走 DynamicSFTCollator,**dataset 行数 = 题数(每题一行)**,
每步从该题的 top-25% 候选池里随机采一个 (sketch, code)。10 epoch 后每题被看
10 次,每次见到不同候选 —— 正则化,不死记单一 (sketch, code)。eval_loss 单调
下降至 plateau,模型可用。

---

## Phase 5:DPO ×4 任务训练

**所有 DPO 从 SFT-ALG-25 的 best 接力**(论文标准 setup),串行跑(每个 ~6-12h)。

### 通用模板(填 $TASK)

```bash
TASK=...   # gvb / pvf / qvs / all
deepspeed --num_gpus 2 --module v2.scripts.07_train_dpo \
    --train_merged $DATASETS/train/merged.jsonl \
    --val_merged $DATASETS/val/merged.jsonl \
    --model_path $RUNS/sft_alg_top25/best \
    --output_dir $RUNS/dpo_$TASK \
    --task $TASK --augment True \
    --num_train_epochs 50 \
    --pairs_per_problem 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --ds_config v2/configs/ds_zero2_2gpu_offload.json
```

⚠️ **`--pairs_per_problem 10` 必须显式覆盖** —— 默认 K=num_train_epochs,
传 `--num_train_epochs 50` 会把 K 一起拉到 50,导致 effective_E 还是 1,只在
末尾存 1 个 ckpt,没法看曲线、没法选中间最佳 ckpt。固定 K=10 让 effective_E=5,
**总训练量同等**(94×10×5 = 4700 pair-visits),但能看 5 个 epoch 的 trajectory。

⚠️ `--num_train_epochs 50` 是关键覆盖:默认 10 + 默认 K=10 联动出 effective_E=1,
对小数据(94 GvB 题)只有 59 步,严重欠拟合(loss=4.3, grad_norm=217)。

⚠️ `--per_device_eval_batch_size 1` 同 SFT,必须显式覆盖(默认 2 会 OOM)。

### 跑哪个就把上方的 `$TASK` 填实

- **5a `dpo_gvb`** —— headline(algo-quality 偏好)
- **5b `dpo_pvf`** —— baseline(binary 正确性偏好)
- **5c `dpo_qvs`** —— timing(全 pass 内快慢偏好)
- **5d `dpo_all`** —— fallback(pvf → qvs → gvb 随机)

### ✋ 核查点 5(每个 DPO 跑完都做一次)

```bash
TASK=...
ls -la $RUNS/dpo_$TASK/best
python -c "
import json
s = json.load(open('$RUNS/dpo_$TASK/trainer_state.json'))
print('best:', s.get('best_model_checkpoint'), 'metric:', s.get('best_metric'))
print('--- training curve ---')
for r in s['log_history']:
    if 'rewards/margins' in r:
        print(f\"step={r['step']:4d} loss={r.get('loss',0):.3f} \"
              f\"margins={r['rewards/margins']:+.2f} \"
              f\"acc={r['rewards/accuracies']:.3f}\")
"
```

健康标志:
- [ ] `rewards/margins` 整体为正、随训练增大
- [ ] `rewards/accuracies` 升到 0.7+
- [ ] `eval_loss` 不发散
- [ ] `best/` symlink 存在

### 结果日志

```
dpo_gvb best ckpt / eval_loss = checkpoint-236 / 0.5203  (margins=+0.60, 单调收敛 ✓ headline)
dpo_pvf best ckpt / eval_loss = checkpoint-166 / 0.7041  (margins=+0.029, ep1 peak)
dpo_qvs best ckpt / eval_loss = checkpoint-345 / 0.6153  (margins=+0.38, ep3 peak,ep4-5 反弹)
dpo_all best ckpt / eval_loss = checkpoint-167 / 0.7094  (margins=-0.004, eval_loss 0.71→1.23 恶化 ✗ 负面对照)
```

观察:GvB >> QvS > PvF > ALL(val 阶段)。ALL 退化是预期 —— 混合 PvF/QvS/GvB 信号
互相打架,且被 yieldable 最多的 PvF 主导。这正好支撑论文 thesis: focused signal
> mixed signal。

---

## Phase 6:5 模型评测

逐个跑(也可后台串起来):

```bash
for run in sft_alg_top25 dpo_gvb dpo_pvf dpo_qvs dpo_all; do
    echo "=== eval $run ==="
    bash v2/scripts/dp_eval.sh $EVALS/$run \
        --problems_jsonl $APPS/test.jsonl \
        --model_path $RUNS/$run/best \
        --n_per_temp 100 --temps 0.6 \
        --do_timing
done
```

每个 ~1-2h × 5 = 5-10h。

### ✋ 核查点 6(最终对比)

```bash
echo "model              pass@1   pass@10  mean_rt(ms)"
for run in base sft_alg_top25 dpo_gvb dpo_pvf dpo_qvs dpo_all; do
    python -c "
import json
m = json.load(open('$EVALS/$run/metrics.json'))
print(f'{\"$run\":18s} {m.get(\"pass@1\",0):.3f}   '
      f'{m.get(\"pass@10\",0):.3f}   '
      f'{m.get(\"mean_runtime_ns\",0)/1e6:.1f}')"
done
```

期望(论文成立):
- [ ] 所有训完模型 pass@1 > BASE
- [ ] **DPO-GvB > DPO-PvF** in pass@10(headline 命题)
- [ ] DPO-QvS mean_runtime < SFT-ALG-25(QvS 卖点)

### 最终结果矩阵(待回填)

```
model              pass@1   pass@10  mean_rt(ms)
base                _.___    _.___    _____
sft_alg_top25       _.___    _.___    _____
dpo_gvb             _.___    _.___    _____
dpo_pvf             _.___    _.___    _____
dpo_qvs             _.___    _.___    _____
dpo_all             _.___    _.___    _____
```

---

## 失败重试策略

| 阶段 | 中断后 |
|------|--------|
| 采样(Phase 0/1/6) | append-only,直接重跑同条命令,会跳过已完成 (task_id, sample_id, code_id) |
| 标注(Phase 2) | 验证 scores.jsonl 是否支持续传;不支持就清空重跑或加 `--resume` 改造 |
| 训练(Phase 4/5) | DeepSpeed 自动从最后 checkpoint 续传,`--output_dir` 不变即可 |

## 关键 ENV 复查表

每次开新 phase 前先 `env | grep -E "WORK|APPS|GLM|PYTORCH_CUDA"` 确认没 unset。

## 跑歪了的硬规则

- 改算法/超参 → 必须在本文件记一笔(改动 + 原因)
- 任何阶段 OOM → 调 batch_size / grad_accum,不改研究语义参数
- GLM 评分明显偏(全 0 或全 10)→ 暂停,检查 prompt 或 alpha
- DPO 训完 `rewards/margins ≤ 0` → 数据集质量问题,先排查 merge / 标注
