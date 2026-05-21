#!/usr/bin/env bash
# Data-parallel 双卡 eval launcher(配合 5090 ×2)。
#
# 与 dp_sample.sh 同构:两个进程各占一卡跑 v2.evaluation.eval_sampling
# 的一半 problems,跑完合并 jsonl,然后跑一次 v2.evaluation.metrics 算指标。
#
# 用法:
#   bash v2/scripts/dp_eval.sh /data/work/out/evals/dpo_gvb \
#       --problems_jsonl /data/work/out/apps/test.jsonl \
#       --model_path /data/work/out/runs/dpo_gvb \
#       --n_per_temp 100 --temps 0.6 --do_timing

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <OUT_DIR> [args forwarded to eval_sampling...]" >&2
    exit 1
fi

OUT_DIR="$1"; shift
SHARD0_DIR="${OUT_DIR}/_shard0"
SHARD1_DIR="${OUT_DIR}/_shard1"
mkdir -p "$SHARD0_DIR" "$SHARD1_DIR"

for arg in "$@"; do
    case "$arg" in
        --out_dir|--shard_id|--n_shards)
            echo "[dp_eval] forbidden forwarded arg: $arg (launcher 内部已注入)" >&2
            exit 2
            ;;
    esac
done

# 同 dp_sample.sh:两 shard 各占自己的 torch.compile / triton 缓存子目录,
# 避免同时编译同一份 graph 时抢同一文件造成 truncated-pickle crash。
TORCHINDUCTOR_BASE="${TORCHINDUCTOR_CACHE_DIR:-/data/cache/torchinductor}"
TRITON_BASE="${TRITON_CACHE_DIR:-/data/cache/triton}"
mkdir -p "$TORCHINDUCTOR_BASE/shard0" "$TORCHINDUCTOR_BASE/shard1" \
         "$TRITON_BASE/shard0" "$TRITON_BASE/shard1"

echo "[dp_eval] launching 2 shards into $OUT_DIR"

CUDA_VISIBLE_DEVICES=0 \
TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_BASE/shard0" \
TRITON_CACHE_DIR="$TRITON_BASE/shard0" \
python -m v2.evaluation.eval_sampling \
    --out_dir "$SHARD0_DIR" --shard_id 0 --n_shards 2 "$@" \
    > "$SHARD0_DIR/stdout.log" 2> "$SHARD0_DIR/stderr.log" &
PID0=$!

CUDA_VISIBLE_DEVICES=1 \
TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_BASE/shard1" \
TRITON_CACHE_DIR="$TRITON_BASE/shard1" \
python -m v2.evaluation.eval_sampling \
    --out_dir "$SHARD1_DIR" --shard_id 1 --n_shards 2 "$@" \
    > "$SHARD1_DIR/stdout.log" 2> "$SHARD1_DIR/stderr.log" &
PID1=$!

echo "[dp_eval] shard0 pid=$PID0, shard1 pid=$PID1"

FAILED=0
wait $PID0 || FAILED=1
wait $PID1 || FAILED=1
if [[ $FAILED -ne 0 ]]; then
    echo "[dp_eval] one or both shards failed" >&2
    exit 3
fi

echo "[dp_eval] merging shard outputs"
for fname in sketches.jsonl codes.jsonl exec.jsonl; do
    cat "$SHARD0_DIR/$fname" "$SHARD1_DIR/$fname" > "$OUT_DIR/$fname"
done

echo "[dp_eval] computing metrics"
python -m v2.evaluation.metrics \
    --exec_path "$OUT_DIR/exec.jsonl" \
    --out "$OUT_DIR/metrics.json"

echo "[dp_eval] done. Outputs:"
ls -la "$OUT_DIR"/*.jsonl "$OUT_DIR/metrics.json"
