"""
两阶段采样的统一执行器。

`sample_sketches(backend, problems, n_per_temp, temps, prompt_token_limit, out_path)`:
    Stage-1。每题对每个温度采 n_per_temp 个 sketch。
    跳过 prompt 过长(prompt_token_limit)的题。
    增量写 sketches.jsonl,断点续跑。

`sample_codes(backend, sketches, problems, prompt_token_limit, out_path)`:
    Stage-2。对每个解析成功的 sketch,采 1 个 code。
    增量写 codes.jsonl,断点续跑。
"""
from __future__ import annotations
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..data.schema import Problem, SketchSample, CodeSample
from ..data.prompts import build_sketch_prompt, build_code_prompt
from .backend import Backend, GenerationConfig
from .parser import parse_sketch, parse_code


log = logging.getLogger(__name__)


# ============================================================
# 工具
# ============================================================

def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _load_done_keys(path: str, key_fn) -> set:
    """读取已写出的 jsonl,返回已完成的 key 集合,用于断点续跑。"""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(key_fn(obj))
            except Exception:
                continue
    return done


def _append_jsonl(path: str, objs: Iterable[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


# ============================================================
# Stage-1: 采样 sketch
# ============================================================

def sample_sketches(
    backend: Backend,
    problems: list[Problem],
    n_per_temp: int,
    temps: list[float],
    out_path: str,
    tokenizer=None,
    prompt_token_limit: int = 2048,
    max_new_tokens: int = 256,
    seed: int = 1,
) -> None:
    """
    对每题在每个温度下采 n_per_temp 个 sketch。结果以 SketchSample 落盘到 out_path。
    sample_id 在每题内全局唯一(跨温度连续编号)。
    """
    done = _load_done_keys(out_path, lambda o: (o["task_id"], o["sample_id"]))
    if done:
        log.info(f"Resuming sketch sampling: {len(done)} samples already done.")

    stop = ["</SKETCH>", "\n### "]

    for problem in problems:
        prompt = build_sketch_prompt(problem)
        if tokenizer is not None and _count_tokens(tokenizer, prompt) > prompt_token_limit:
            log.warning(f"Skipping {problem.task_id}: prompt too long.")
            continue

        # 已完成数
        already = sum(1 for (t, _) in [k for k in done if k[0] == problem.task_id])
        sample_id_base = 0
        new_objs: list[dict] = []

        for ti, temp in enumerate(temps):
            need = n_per_temp
            # 跳过已完成的(简单按数量,假定写入顺序稳定)
            start_id = sample_id_base
            sample_id_base += n_per_temp
            ids_for_this_temp = list(range(start_id, start_id + need))
            ids_to_run = [i for i in ids_for_this_temp if (problem.task_id, i) not in done]
            if not ids_to_run:
                continue

            cfg = GenerationConfig(
                n_per_prompt=len(ids_to_run),
                temperature=temp,
                max_new_tokens=max_new_tokens,
                stop=stop,
                seed=seed + ti,
            )
            completions = backend.generate([prompt], cfg)[0]
            assert len(completions) == len(ids_to_run)

            for sid, completion in zip(ids_to_run, completions):
                sketch_text, ok = parse_sketch(completion)
                s = SketchSample(
                    task_id=problem.task_id,
                    sample_id=sid,
                    sketch=sketch_text,
                    sample_temp=temp,
                    raw_completion=completion,
                    parsed_ok=ok,
                )
                new_objs.append(asdict(s))

        if new_objs:
            _append_jsonl(out_path, new_objs)
            for o in new_objs:
                done.add((o["task_id"], o["sample_id"]))


# ============================================================
# Stage-2: 采样 code(每个有效 sketch 配 1 个 code)
# ============================================================

def sample_codes(
    backend: Backend,
    problems_by_id: dict[str, Problem],
    sketches_path: str,
    out_path: str,
    tokenizer=None,
    prompt_token_limit: int = 3072,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    seed: int = 1,
) -> None:
    """
    读 sketches_path 里 parsed_ok=True 的 SketchSample,
    为每个 sketch 采 1 个 code,写到 out_path。
    """
    # 读所有 sketch
    sketches: list[dict] = []
    with open(sketches_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("parsed_ok"):
                sketches.append(obj)

    done = _load_done_keys(out_path, lambda o: (o["task_id"], o["sample_id"], o.get("code_id", 0)))
    log.info(f"Code sampling: {len(sketches)} valid sketches, {len(done)} already done.")

    stop = ["\n```", "\n### "]
    BATCH = 32  # 把多个 prompt 攒一批给 vllm,提速
    pending: list[tuple[dict, str]] = []   # (sketch_obj, prompt)

    def _flush():
        if not pending:
            return
        prompts = [pr for _, pr in pending]
        cfg = GenerationConfig(
            n_per_prompt=1, temperature=temperature,
            max_new_tokens=max_new_tokens, stop=stop, seed=seed,
        )
        outs = backend.generate(prompts, cfg)
        new_objs = []
        for (s_obj, _), completions in zip(pending, outs):
            completion = completions[0]
            code_text, ok = parse_code(completion)
            c = CodeSample(
                task_id=s_obj["task_id"],
                sample_id=s_obj["sample_id"],
                code_id=0,
                sketch=s_obj["sketch"],
                code=code_text,
                sample_temp=s_obj["sample_temp"],
                raw_completion=completion,
                parsed_ok=ok,
            )
            new_objs.append(asdict(c))
        _append_jsonl(out_path, new_objs)
        for o in new_objs:
            done.add((o["task_id"], o["sample_id"], 0))
        pending.clear()

    for s_obj in sketches:
        key = (s_obj["task_id"], s_obj["sample_id"], 0)
        if key in done:
            continue
        problem = problems_by_id.get(s_obj["task_id"])
        if problem is None:
            continue
        prompt = build_code_prompt(problem, s_obj["sketch"])
        if tokenizer is not None and _count_tokens(tokenizer, prompt) > prompt_token_limit:
            continue
        pending.append((s_obj, prompt))
        if len(pending) >= BATCH:
            _flush()
    _flush()
