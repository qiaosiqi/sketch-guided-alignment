# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The actual project lives in the `sketch-guided-alignment/` subdirectory. The repo root also contains `environment.yml` (full pinned conda env, name `hebo_env`, Python 3.10) and a `论文` directory with the reference paper. All commands below assume the working directory is `sketch-guided-alignment/`.

This is a fork/restructuring of **Code-Optimise** (Gee et al., 2024, arXiv:2406.12502): self-generated preference data for optimising code LMs on functional correctness and runtime efficiency.

## Pipeline (the big picture)

The codebase implements a four-stage loop. Each stage is a shell script that prompts for parameters via `read -p` and then invokes a Python script under `torchrun` with DeepSpeed. Paths in the `.sh` scripts contain a hardcoded `/YOUR_BASE_PATH/` prefix that must be substituted before running.

1. **Sample** (`sample.sh` → `infer.py --eval_mode False --regen True`): generate N candidate completions per training problem with the base model (e.g. StarCoder-1B, CodeLlama). Writes `samples/<save_path>/<data_path>/<split>/outputs.json`.
2. **Annotate** (`sample.sh` → `infer.py --eval_mode False --regen False`): re-run with `--regen False` over the same `save_path` to execute each completion 5 times against the unit tests in `utils/execution.py`, recording pass/fail and per-sample runtime. Results go into `runs/check_{0..4}.json`.
3. **Merge & filter** (`merge.sh` → `merge.py`): aggregate the 5 runs, dedupe completions, then keep only problems that have **≥2 passing and ≥1 failing** solutions (this is the filter that makes pairwise DPO data possible). Output: `datasets/<save_path>/<split>/merged.json` with each row containing `question` + a list of `answers` (each `{text, votes}` where `votes = pass ? avg_time : inf`).
4. **Optimise** (`train.sh` → `train.py`): SFT or DPO on the merged data.
5. **Evaluate** (`eval.sh` → `infer.py --eval_mode True`): same two-pass (`--regen True` then `--regen False`) but on the test split of MBPP / HumanEval. Outputs pass@k and avg runtime to `tests<save_path>/<data_path>/results.txt`.

The "preference" signal is **runtime among passing solutions**: faster passes are preferred. `votes = inf` for any failing solution. This is the value of `votes` consumed by `keep_top` / `dpo_format` in `utils/collator.py`.

## Critical timing caveat

`utils/execution.py` re-runs each completion up to 1000 times in batches of 50 until `std/avg ≤ 0.1` (coefficient of variation), to get stable per-sample runtimes. This makes the annotate step **wall-clock sensitive**: per the README, always use the same machine, no other heavy processes, and don't run htop. If you change anything in `execution.py` (timeout, repeat counts, the CoV threshold), you invalidate cross-run comparisons.

## Training modes (`train.py`)

- **SFT** (`--optim sft --top_p N`): `keep_top` selects the top-N% of answers by `votes` (fastest passing first). With `--augment True`, the SFT collator samples one of the top-N each step (dynamic); `--augment False` keeps a fixed one. `SFTCollator` randomly picks one answer per example per batch in `torch_call`.
- **DPO** (`--optim dpo --task {qvs,pvf,all}`):
  - `qvs` — quick-vs-slow: chosen and rejected both pass; chosen is faster.
  - `pvf` — pass-vs-fail: chosen passes, rejected fails.
  - `all` — chosen is some passing solution, rejected is anything ranked below it (passing-but-slower or failing).
  With `--augment True`, the pair is re-sampled per step inside `DPOCollator` (via `dpo_format` in the collator); `--augment False` materialises the pair once via `Dataset.map`.

`TrainingArguments` are hard-coded in `train.py:89-111` (30 epochs, lr 5e-7, batch 2 × grad-accum 16, fp16, eval/save per epoch, load best on `eval_loss`). Edit there to change them.

The `### Question:` / `### Answer:` template (`sft_format`) and the response-template tokenisation (`train.py:140-142`) have a Llama-specific tweak — the first token is dropped for `model_type == 'llama'`. Be careful when adding new model families.

## Environment

- `requirements.txt` is the upstream pin set (Python 3.10, `transformers==4.35.0`, `trl==0.7.4`, `deepspeed==0.12.3`, `torch==2.1.0`, `datasets==2.14.6`). `trl` 0.7.x is required because the code subclasses `trl.SFTTrainer` and `trl.trainer.utils.DPODataCollatorWithPadding` — newer trl versions reorganised both.
- `environment.yml` is a fuller conda dump for the same Python 3.10 stack on Linux (CUDA 12.1 wheels). The `.sh` scripts assume a bash/Linux shell with `torchrun` and CUDA; on Windows use WSL or run the Python scripts directly.
- `ds_zero.json` is DeepSpeed ZeRO-3 with CPU optimizer + param offload. Used by `train.py` only; `infer.py` uses `deepspeed.init_inference` with tensor-parallel sharding across `torch.cuda.device_count()`.

## Common commands

The shell scripts are interactive (`read -p`); to run non-interactively, pipe answers in or call the Python entry points directly. Examples (substitute paths/devices):

```bash
# Sample 100 completions per problem at temp 0.6 with StarCoder-1B on GPU 0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 infer.py \
  --eval_mode False --regen True \
  --data_path .../datasets/mbpp --model_path .../models/StarCoder-1B \
  --n_seq 100 --n_iter 1 --sample True --temp 0.6 \
  --save_path .../samples/sc-1b-ms-0.6/mbpp --seed 1

# Annotate (same save_path, --regen False), then merge
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 infer.py \
  --eval_mode False --regen False --data_path .../datasets/mbpp \
  --save_path .../samples/sc-1b-ms-0.6/mbpp --seed 1
torchrun merge.py --data_path .../samples/sc-1b-ms-0.6 --save_path .../datasets/sc-1b-ms-0.6

# Train DPO with quick-vs-slow pairs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 train.py \
  --data_path .../datasets/sc-1b-ms-0.6/mbpp --model_path .../models/StarCoder-1B \
  --optim dpo --task qvs --augment True \
  --ds_config ds_zero.json --save_path .../models/mbpp-1b-1b-dpo-qvs --seed 1
```

There are no tests, linter config, or build step in this repo.

## Datasets

- Training/validation: MBPP train+validation splits, loaded via `datasets.load_from_disk` (an `arrow`/HF-saved directory keyed by `task_id`, `prompt`, `test`, `entry_point`).
- Evaluation: MBPP test + HumanEval.
- `datasets/APPS/raw/` contains a sample of APPS problems (`question.txt`, `solutions.json`, `input_output.json`, optional `starter_code.py`) — this is not directly consumed by the pipeline as written; `infer.py` expects a `load_from_disk` directory.

The merge step asserts every problem in the split has at least one attempted sample (`assert len(completion_id) == len(problems)` in `utils/evaluation.py`); if you shard the sampling step, you must concatenate before annotating.
