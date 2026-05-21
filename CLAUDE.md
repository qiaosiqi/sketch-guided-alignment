# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Repository layout

- **`v2/`** — the active research codebase (the one being written up as a paper). All work happens here.
- `sketch-guided-alignment/` — the upstream baseline (a fork of Code-Optimise / Gee et al., 2024, arXiv:2406.12502) kept for reference only. Do not edit it.
- `environment.yml` — upstream baseline pinned env (`hebo_env`, torch 2.1 + CUDA 12.1). Reference only; will not work on Blackwell.
- `v2/scripts/setup_env.sh` — phased install of the active env (`sketch5090`, torch 2.7+cu128, vllm 0.8.5, trl ≥ 0.11). Only the three Blackwell-critical packages are pinned; the rest are resolved by pip.
- `论文/` — reference paper drafts.

All commands and path references below assume the working directory is `v2/`.

## Pipeline overview

Eight numbered entry points in `scripts/` run in order:

| Script | Purpose |
|--------|---------|
| `01_prepare_apps.py` | Load APPS interview-only problems → `train/val/test.jsonl` (9:1 train split) |
| `02_sample_pilot.py` | Two-stage sampling: sketch candidates → one code per sketch |
| `03_analyze_pilot.py` | Analyze pilot outputs to set main-run hyperparameters |
| `04_annotate.py` | GLM-4-Air 9D quality scoring of (sketch, code) pairs |
| `05_merge.py` | Merge execution results + scores → `merged.jsonl` |
| `06_train_sft.py` | SFT on top-p% of answers |
| `07_train_dpo.py` | DPO with one of five preference-pair tasks |
| `08_eval.py` | Test-set sampling → execution → `pass@k` / runtime / algo-score metrics |

## Algorithm constraints

These are the core design choices that define the research contribution; do not silently change them.

**Two-stage generation (sketch → code)**
Every completion is produced in two steps: (1) generate a brief algorithmic sketch (2–4 sentences, extracted from `<SKETCH>` tags), then (2) generate code conditioned on that sketch. Direct code generation without a sketch violates the experiment design.

**Partial credit execution**
`pass_ratio = n_passed_tests / n_total_tests ∈ [0, 1]`. Runtime is measured **only** when `pass_ratio == 1.0`. Using a binary pass/fail or measuring runtime for partial-pass solutions breaks the scoring invariant.

**9-dimensional rubric scoring (via GLM-4-Air)**
Four sketch dimensions (S1–S4) + five code dimensions (C1–C5), all integers 0–10.
Final score: `algo_final = 0.4 · mean(S1..S4) + 0.6 · mean(C1..C5)`.
The code scoring call includes the sketch as reference (faithfulness is C1, a key dimension).

**DPO preference tasks**
Four named tasks; each defines how chosen/rejected pairs are formed from a problem's answer pool:

| Task | Chosen | Rejected |
|------|--------|---------|
| `pvf` | `pass_ratio == 1.0` | `pass_ratio == 0.0` |
| `qvs` | fastest (full-pass only) | slowest (full-pass only) |
| `gvb` | `algo_final ≥ 6.0` (with `pass_ratio ≥ 0.5`) | `algo_final < 6.0` (with `pass_ratio ≥ 0.5`) |
| `all` | tries pvf → qvs → gvb in random order; uses first success | — |

`gvb` is the headline contribution (algorithm-quality signal independent of pass_ratio). `pvf` is the binary correctness signal. The original design also included an `hvl` task (`pass_ratio ≥ 0.7` vs `≤ 0.3`); it was removed on 2026-05-21 after the pilot showed StarCoder2-3B × APPS interview pass_ratio is near-bimodal (92% = 0, 4.6% = 1), making HvL identical to PvF. Do not re-introduce HvL without a base model that fills the partial-credit region.

**SFT selection**
Top-p% of answers per problem, ranked by `algo_final` / `pass_ratio` / runtime. Dynamic selection (re-sampling per batch step) is the default mode.

## Flow constraints

- **Data schema** is the contract between stages; defined in `data/schema.py`. Inter-stage data must conform to `Problem`, `SketchSample`, `CodeSample`, `ExecutionResult`, `AlgorithmScores`, `MergedAnswer`, `MergedProblem`.
- **Execution isolation runs on Linux/macOS only** (SIGALRM-based sandbox). The sandbox disables dangerous builtins; do not weaken it.
- **Stable timing**: repeat executions until runtime coefficient of variation ≤ 0.1. Changing the CoV threshold invalidates cross-run comparisons.
- **Dataset scope**: APPS **interview-only** problems. Competition and introductory difficulties are excluded.
- **Problems with zero valid answers** are dropped at merge; downstream training scripts must not receive problems with empty answer arrays.
- **Annotation is concurrent**: `04_annotate.py` fires judge calls concurrently with rate-limit backoff. The judge must be GLM-4-Air (set `GLM_API_KEY` env var); do not swap in a different model without updating the rubric calibration.

## Environment

- **Hardware target**: 2 × NVIDIA RTX 5090 32G (Blackwell sm_120), bf16. SFT: DeepSpeed ZeRO-3 (`ds_zero3_2gpu.json`) — ZeRO-2 hits OOM on Adam state init for a 3B model on 32G cards. DPO: DeepSpeed ZeRO-2 + CPU optimizer offload (`ds_zero2_2gpu_offload.json`) **plus** `precompute_ref_log_probs=True` — TRL forbids precompute under ZeRO-3, so we use ZeRO-2 and offload Adam state to the 240GB host RAM to free GPU. Precompute then drops the 6GB reference model from GPU after init. All training commands also need `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce 5090 fragmentation pressure.
- **Base model**: StarCoder2-3B (default; paths are config-driven).
- **Judge model**: GLM-4-Air via ZhipuAI HTTP API.
- **Sampling backend**: vLLM (primary) with HF Transformers fallback.
- **Key library versions**: Python 3.10, trl ≥ 0.11 (unlike the upstream baseline which required trl 0.7.x), DeepSpeed, vLLM.
