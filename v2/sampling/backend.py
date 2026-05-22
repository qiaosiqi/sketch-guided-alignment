"""
采样后端抽象。两套实现:
    - vLLMBackend:基于 vllm.LLM,batched 高吞吐,V100 + StarCoder2 通常可用
    - HFBackend:基于 transformers.AutoModelForCausalLM.generate,保底 fallback

统一接口:
    backend = build_backend(model_path, dtype="float16", prefer="vllm")
    completions = backend.generate(prompts, n_per_prompt, temperature, max_new_tokens, stop)
        # → list[list[str]],外层长度 = len(prompts),内层长度 = n_per_prompt

设计要点:
- prefer="vllm" 时先 try vllm,装不上或初始化失败就 log warning 回退到 HF
- prefer="hf" 时直接走 transformers
- 不再像旧代码那样用 deepspeed inference(对小模型反而拖后腿,且与 vllm 互斥)
- stop tokens:sketch 阶段用 ["</SKETCH>", "\n###"];code 阶段用 ["```\n", "\n###"]
"""
from __future__ import annotations
import logging
import os
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


log = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    n_per_prompt: int = 1
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 1024
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


# ============================================================
# 抽象基类
# ============================================================

class Backend(ABC):
    name: str = "abstract"

    @abstractmethod
    def generate(self, prompts: List[str], cfg: GenerationConfig) -> List[List[str]]:
        """返回 [n_prompts][n_per_prompt] 的字符串矩阵。"""

    def shutdown(self):
        """释放资源(GPU 显存等)。默认无操作。"""
        pass


# ============================================================
# vLLM 后端
# ============================================================

class VLLMBackend(Backend):
    name = "vllm"

    def __init__(
        self,
        model_path: str,
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        seed: int = 1,
    ):
        from vllm import LLM  # type: ignore
        log.info(f"Loading vLLM backend: {model_path} dtype={dtype}")
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            seed=seed,
        )

    def generate(self, prompts: List[str], cfg: GenerationConfig) -> List[List[str]]:
        from vllm import SamplingParams  # type: ignore
        sp = SamplingParams(
            n=cfg.n_per_prompt,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            stop=cfg.stop or [],
            seed=cfg.seed,
        )
        outputs = self.llm.generate(prompts, sp, use_tqdm=False)
        # outputs[i].outputs 是 n_per_prompt 个 CompletionOutput
        return [[o.text for o in resp.outputs] for resp in outputs]

    def shutdown(self):
        try:
            import gc
            import torch
            # destroy_model_parallel 清理 vLLM 起的 NCCL 进程组 + EngineCore 后台进程
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass
            del self.llm
            gc.collect()
            torch.cuda.empty_cache()
            # synchronize 确保所有 CUDA op 完成再 fork,
            # 减少子进程继承"进行中的 CUDA 操作"的概率
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


# ============================================================
# transformers 后端(fallback)
# ============================================================

class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_path: str, dtype: str = "float16", device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info(f"Loading HF backend: {model_path} dtype={dtype} device={device}")
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=True,
        ).to(device).eval()
        self.device = device

    def generate(self, prompts: List[str], cfg: GenerationConfig) -> List[List[str]]:
        import torch
        results: List[List[str]] = []
        # HF generate 一次只能给一个 prompt 拿 n 个返回,所以外层循环 prompt
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    do_sample=(cfg.temperature > 0),
                    temperature=max(cfg.temperature, 1e-5),
                    top_p=cfg.top_p,
                    max_new_tokens=cfg.max_new_tokens,
                    num_return_sequences=cfg.n_per_prompt,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            completions = []
            prompt_len = enc["input_ids"].shape[1]
            for seq in out:
                text = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                # 手动应用 stop 序列
                for s in (cfg.stop or []):
                    idx = text.find(s)
                    if idx != -1:
                        text = text[:idx]
                completions.append(text)
            results.append(completions)
        return results

    def shutdown(self):
        try:
            import torch
            del self.model
            torch.cuda.empty_cache()
        except Exception:
            pass


# ============================================================
# 工厂
# ============================================================

def build_backend(
    model_path: str,
    dtype: str = "float16",
    prefer: str = "vllm",
    allow_fallback: bool = False,
    **kwargs,
) -> Backend:
    """构造采样后端。

    `prefer="vllm"` 默认是 strict 的:vllm 起不来直接 raise,把 traceback 打全。
    之所以默认严格,是吃过一次亏 —— 静默回退到 HF 会让性能掉 5–10×,
    用户从日志里只看到一行 warning,等发现时已经浪费几个小时。
    确实想容错就显式 `allow_fallback=True`(或干脆 `prefer="hf"`)。
    """
    if prefer == "vllm":
        try:
            return VLLMBackend(model_path, dtype=dtype, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            if not allow_fallback:
                log.error(f"vLLM backend failed and allow_fallback=False:\n{tb}")
                raise RuntimeError(
                    f"vLLM backend failed: {e!r}. "
                    "Pass allow_fallback=True or prefer='hf' if degradation is acceptable."
                ) from e
            log.warning(f"vLLM backend failed; falling back to HF.\nTraceback:\n{tb}")
    return HFBackend(model_path, dtype=dtype)
