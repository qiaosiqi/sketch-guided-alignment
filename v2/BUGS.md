# 已知 Bug 记录

## BUG-001: Execution 阶段子进程退出卡顿,导致 exec 速度慢 50-100×

**状态**: ✅ 已修复(2026-05-22)

**发现时间**: 2026-05-22

**现象**:
- `dp_eval.sh --do_timing` 启动后,`run_executions_parallel` 日志输出后 7 分钟内 exec.jsonl 不写任何行,被外部 SIGTERM 杀掉。
- 去掉 `--do_timing` 重跑,execution 缓慢推进但仍极慢:**2 小时仅写出 ~5000 行**(目标 ~300k 行),按此速度全程需 ~120 小时。

**根本原因 —— 子进程继承 CUDA context**:

`v2/evaluation/eval_sampling.py` 在 sampling 结束后调用 `backend.shutdown()`,但
`VLLMBackend.shutdown()` 原实现仅:

```python
del self.llm
torch.cuda.empty_cache()
```

这只清了 PyTorch 的 GPU 缓存池,**没有销毁 CUDA context**,也没有清理 vLLM 起的 NCCL 进程组和后台 EngineCore 进程。

随后 `run_executions_parallel` 用 `ThreadPoolExecutor(max_workers=12)` 调度,每个 worker 通过 `multiprocessing.Process` fork 子进程跑用户代码。**fork 出来的子进程继承了父进程残留的 CUDA context**。

子进程跑完 `_worker` 正常返回时,Python 解释器执行 atexit 链,其中 `torch.cuda` 注册的 cleanup 试图与 CUDA driver 通信清理已损坏(post-fork)的 context,**每个子进程退出耗时 ~30 秒**。

149k codes / 12 workers × 30s ≈ 100+ 小时,即观察到的速度。

**次要问题 —— `hard_wall` 系数过大**:

`v2/execution/runner.py`:

```python
hard_wall = timeout_per_test * len(problem.inputs) * (60 if do_timing else 2) + 10
```

`do_timing=True` + 100 个测试用例:`3 × 100 × 60 + 10 = 18,010 秒 ≈ 5 小时`。

`stable_runtime` 内部已用 `time_limit(timeout_per_test * n_inputs * 2)` 自我约束(对 100 inputs 是 600 秒),外层 hard_wall 没必要再乘 60。叠加上面的 CUDA cleanup 卡顿,`--do_timing` 时 12 个 worker 全部阻塞,`as_completed` 永远拿不到结果。

**影响范围**:
- `v2/scripts/dp_eval.sh`(任何模型评测都会卡)
- `v2/scripts/dp_sample.sh`(主采样 train+val 同样受影响)
- `v2/scripts/02_sample_pilot.py`、`v2/evaluation/eval_sampling.py`

**修复**:

1. **`v2/execution/runner.py` — 子进程 `os._exit(0)` 跳过 atexit**:
   ```python
   result_pipe.send({...})
   os._exit(0)  # 跳过 torch/CUDA atexit cleanup
   ```
   `os._exit` 不在 `reliability_guard()` 黑名单内,可安全调用。子进程跳过 Python
   cleanup 直接退出,内核回收文件描述符和虚拟内存,CUDA context 也随之释放。

2. **`v2/sampling/backend.py` — 加强 `VLLMBackend.shutdown()`**:
   ```python
   from vllm.distributed.parallel_state import destroy_model_parallel
   destroy_model_parallel()   # 清理 NCCL 进程组 + EngineCore 后台进程
   del self.llm
   gc.collect()
   torch.cuda.empty_cache()
   torch.cuda.synchronize()   # 等所有 CUDA op 完成再 fork
   ```

3. **`v2/execution/runner.py` — hard_wall 系数 60 → 10**:
   ```python
   hard_wall = timeout_per_test * len(problem.inputs) * (10 if do_timing else 2) + 10
   ```
   配合 `stable_runtime` 内层 `time_limit`,100-input 题目最多等 3010 秒,合理。

**遗留风险**:

- 子进程 `os._exit(0)` 跳过 `tempfile.TemporaryDirectory.__exit__`,可能在
  `/tmp` 累积空目录。`reliability_guard` 已禁用 `shutil.rmtree`,清理本来就不可靠。
  建议主跑期间观察 `df /tmp`,必要时手动清理 `/tmp/tmp*` 模式的旧目录。

---

## BUG-002: 子进程缺 RLIMIT_AS 内存墙,坏解触发 OOM-killer

**状态**: ✅ 已修复(2026-05-22)

**发现时间**: 2026-05-22(BUG-001 修复后立刻暴露)

**现象**:
BUG-001 修复后重跑 BASE eval。GPU 正常释放(498MiB/卡),CPU 满载 22 核,但
**总内存在 150-200GB 波动**(机器 240GB),`watch wc -l` 看 exec.jsonl 数百行
后被 SIGTERM 杀。`dmesg` 不能读但症状高度匹配 OOM-killer。

**根本原因**:

`v2/execution/sandbox.py` 的 `reliability_guard()` 没设进程内存上限。BASE
StarCoder2-3B 经常会生成指数级内存开销的解 —— 比如:

```python
n = int(input())
arr = [[0] * n for _ in range(n)]   # n 是用户输入,可能 10^6,导致 8 TB 分配
```

24 个 worker 并发(每 shard 12)只要其中一个 worker 跑到这种解,瞬间吃几十 GB。
几个并发就把 240GB host RAM 打满,OOM-killer 杀掉整个 shard 进程。

BUG-001 修复前没暴露,是因为子进程退出慢(每个 30s atexit),并发度被自然限流,
峰值内存不高。修复后子进程秒退,并发度真正打到 12,bad solutions 的内存峰值集中
出现 → OOM。

上游 `human-eval` 的 `reliability_guard` 默认就设 `RLIMIT_AS` 4GB,我们之前
照搬时漏了这一段。

**修复**:

`v2/execution/sandbox.py` — 在 nuke `sys.modules["resource"]` **之前**设 rlimit:

```python
def reliability_guard(maximum_memory_bytes: int = 4 * 1024 ** 3):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
    except (ValueError, OSError, ImportError):
        pass
    ...
```

4 GiB 上限:APPS 解几乎用不到这么多,违规者直接 `MemoryError`(被 `_worker`
里 `except BaseException` 捕获标 fail),不影响合法解。24 worker × 4 GiB = 96 GB
最大消耗,远低于 240 GB host。

**为什么不设 RLIMIT_STACK**:某些 APPS 解递归较深,栈墙太低会误杀正确解。Python
默认栈对 APPS 体量足够。

**关联**: 必须和 BUG-001 修复一起部署,否则 BUG-001 修后就会触发 BUG-002。
