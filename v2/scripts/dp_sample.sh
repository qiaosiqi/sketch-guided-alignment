#!/usr/bin/env bash
# Data-parallel 双卡采样 launcher(5090 ×2 / 4090 ×2 通用)。
#
# 起两个进程,每个 pin 到一张卡,分别跑 problems 的偶数和奇数下标分片,
# 跑完把三份产物(sketches/codes/exec)cat 合并到 $OUT_DIR 下。
#
# 用法:
#   bash v2/scripts/dp_sample.sh /data/work/out/sample_train \
#       --problems_jsonl /data/work/out/apps/train.jsonl \
#       --model_path /data/models/StarCoder2-3B \
#       --n_problems 99999 --n_per_temp 100 --temps 0.6
#
# 注意:
# - 第一个位置参数固定为 OUT_DIR(其余参数原样转发给 02_sample_pilot)
# - 不能在转发参数里传 --out_dir / --shard_id / --n_shards,launcher 会自动注入
# - 中断重跑安全:每个 shard 子目录内的 jsonl 都是 append 写,会跳过已完成 (task_id, sample_id, code_id)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <OUT_DIR> [args forwarded to 02_sample_pilot...]" >&2
    exit 1
fi

OUT_DIR="$1"; shift
SHARD0_DIR="${OUT_DIR}/_shard0"
SHARD1_DIR="${OUT_DIR}/_shard1"
mkdir -p "$SHARD0_DIR" "$SHARD1_DIR"

# 拒绝转发与 launcher 内部约定冲突的参数
for arg in "$@"; do
    case "$arg" in
        --out_dir|--shard_id|--n_shards)
            echo "[dp_sample] forbidden forwarded arg: $arg (launcher 内部已注入)" >&2
            exit 2
            ;;
    esac
done

# 两个 shard 同时编译相同的 vllm graph 会抢同一份 torchinductor / triton 缓存,
# 输的一方读到截断 pickle 会直接 crash。给每个 shard 独立 cache 子目录,
# 不动 ~/.bashrc 里的父变量,仅在子进程里拼一层 shard{0,1}。
# 同样原因隔离 VLLM_CACHE_ROOT(~/.cache/vllm/torch_compile_cache 也有竞态,
# atomic-rename 时另一个 shard 清掉同目录,触发 FileNotFoundError → Engine init 失败)。
TORCHINDUCTOR_BASE="${TORCHINDUCTOR_CACHE_DIR:-/data/cache/torchinductor}"
TRITON_BASE="${TRITON_CACHE_DIR:-/data/cache/triton}"
VLLM_BASE="${VLLM_CACHE_ROOT:-/data/cache/vllm}"
mkdir -p "$TORCHINDUCTOR_BASE/shard0" "$TORCHINDUCTOR_BASE/shard1" \
         "$TRITON_BASE/shard0" "$TRITON_BASE/shard1" \
         "$VLLM_BASE/shard0" "$VLLM_BASE/shard1"

echo "[dp_sample] launching 2 shards into $OUT_DIR"

CUDA_VISIBLE_DEVICES=0 \
TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_BASE/shard0" \
TRITON_CACHE_DIR="$TRITON_BASE/shard0" \
VLLM_CACHE_ROOT="$VLLM_BASE/shard0" \
python -m v2.scripts.02_sample_pilot \
    --out_dir "$SHARD0_DIR" --shard_id 0 --n_shards 2 "$@" \
    > "$SHARD0_DIR/stdout.log" 2> "$SHARD0_DIR/stderr.log" &
PID0=$!

CUDA_VISIBLE_DEVICES=1 \
TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_BASE/shard1" \
TRITON_CACHE_DIR="$TRITON_BASE/shard1" \
VLLM_CACHE_ROOT="$VLLM_BASE/shard1" \
python -m v2.scripts.02_sample_pilot \
    --out_dir "$SHARD1_DIR" --shard_id 1 --n_shards 2 "$@" \
    > "$SHARD1_DIR/stdout.log" 2> "$SHARD1_DIR/stderr.log" &
PID1=$!

echo "[dp_sample] shard0 pid=$PID0, shard1 pid=$PID1; tail -f ${SHARD0_DIR}/stderr.log for progress"

# 任何一个失败就让整个 launcher 失败,但要先等另一个收尾,避免半成品
FAILED=0
wait $PID0 || FAILED=1
wait $PID1 || FAILED=1
if [[ $FAILED -ne 0 ]]; then
    echo "[dp_sample] one or both shards failed; logs in $SHARD0_DIR / $SHARD1_DIR" >&2
    exit 3
fi

echo "[dp_sample] merging shard outputs into $OUT_DIR"
for fname in chosen_problems.jsonl sketches.jsonl codes.jsonl exec.jsonl; do
    cat "$SHARD0_DIR/$fname" "$SHARD1_DIR/$fname" > "$OUT_DIR/$fname"
done

echo "[dp_sample] done. Outputs:"
ls -la "$OUT_DIR"/*.jsonl
