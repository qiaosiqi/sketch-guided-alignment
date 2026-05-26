"""
执行沙箱:复用旧 utils/execution.py 的核心安全机制。

仅支持 Linux/macOS(依赖 signal.SIGALRM)。Windows 上做开发时,execution 阶段
请在 WSL 或远程 Linux 跑。

核心导出:
    time_limit(seconds)             — 超时上下文管理器
    swallow_io(stdin_str=None)      — 重定向 stdin/stdout/stderr
    create_tempdir()                — 隔离临时工作目录
    reliability_guard()             — 禁用危险 builtin
    captured_stdout()               — 获取当前 swallow_io 捕获的 stdout
"""
from __future__ import annotations
import contextlib
import faulthandler
import io
import os
import shutil
import signal
import sys
import tempfile


# ============================================================
# 超时
# ============================================================

@contextlib.contextmanager
def time_limit(seconds: float):
    """SIGALRM 实现的硬超时。注意只能在主线程使用,Linux/macOS only。"""
    def handler(signum, frame):
        raise TimeoutError("execution timed out")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


# ============================================================
# IO 重定向
# ============================================================

class _WriteOnlyStringIO(io.StringIO):
    def read(self, *a, **kw): raise IOError
    def readline(self, *a, **kw): raise IOError
    def readlines(self, *a, **kw): raise IOError
    def readable(self, *a, **kw): return False


class _ReadableStringIO(io.StringIO):
    """既能写(被忽略)又能被代码 read/readline 的 stdin。"""
    pass


class _redirect_stdin(contextlib._RedirectStream):  # type: ignore[misc]
    _stream = "stdin"


class _IOHolder:
    """供调用者取回捕获到的 stdout 文本。"""
    def __init__(self):
        self.stdout = ""


@contextlib.contextmanager
def swallow_io(stdin_str: str | None = None):
    """
    重定向 stdin/stdout/stderr。

    - stdin_str is None:stdin 不可读(写代码读了就 IOError),用于无 stdin 的题
    - stdin_str 提供:模拟用户从 stdin 输入,代码可以正常 input() / sys.stdin.read()
    yield 出一个 _IOHolder,退出时把 stdout 内容填进去。
    """
    holder = _IOHolder()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    stdin_buf = _ReadableStringIO(stdin_str) if stdin_str is not None else _WriteOnlyStringIO()

    with contextlib.redirect_stdout(stdout_buf):
        with contextlib.redirect_stderr(stderr_buf):
            with _redirect_stdin(stdin_buf):
                try:
                    yield holder
                finally:
                    holder.stdout = stdout_buf.getvalue()


# ============================================================
# 临时目录
# ============================================================

@contextlib.contextmanager
def create_tempdir():
    # reliability_guard 会把 os.chdir / shutil.rmtree 等置 None。本上下文管理器的
    # 退出阶段(复位 cwd + 清临时目录)必须在 guard 生效之后执行,所以先抓住原引用、
    # 用本地变量调用,避免 cleanup 时拿到 None 触发 TypeError 把整条结果糊成
    # worker_crashed。同理不再用 tempfile.TemporaryDirectory —— 它的 __exit__ 会
    # 调被 nullify 的 shutil.rmtree。
    _chdir_fn = os.chdir
    _rmtree_fn = shutil.rmtree
    cwd = os.getcwd()
    d = tempfile.mkdtemp()
    _chdir_fn(d)
    try:
        yield d
    finally:
        try:
            _chdir_fn(cwd)
        except Exception:
            pass
        try:
            _rmtree_fn(d, ignore_errors=True)
        except Exception:
            pass


# ============================================================
# 反破坏 guard(完全照搬旧版,但保留 subprocess.Popen 引用以便恢复)
# ============================================================

_ORIGINAL = {}


def reliability_guard(maximum_memory_bytes: int = 4 * 1024 ** 3):
    """禁用大量危险 API,并给子进程设硬内存墙。注意会污染当前进程,只在子进程里调用。

    `maximum_memory_bytes` 默认 4 GiB —— APPS 解几乎都用不到这么多,而 BASE 模型
    时常生成指数级开销的解(e.g. `[0] * 10**9`)能瞬间吃满 host RAM。24 个 worker
    并发时只要其中一个不设墙就可能 OOM-kill 整个 shard。上游 human-eval 的
    reliability_guard 默认就有这个,我们之前漏了。

    必须在 `sys.modules["resource"] = None` 之前设 rlimit,否则 resource 模块被
    nuke 之后就回不来了。
    """
    # ⚠️ rlimit 必须最先设,在 resource 模块被 nuke 之前
    try:
        import resource
        # RLIMIT_AS 限制整个进程的虚拟内存(含 mmap),最有效
        # RLIMIT_DATA 限制 heap 大小,作为兜底
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        # 不设 RLIMIT_STACK:某些 APPS 解递归较深,栈不够会误杀正确解
    except (ValueError, OSError, ImportError):
        pass

    faulthandler.disable()

    import builtins
    _ORIGINAL["builtins.exit"] = builtins.exit
    _ORIGINAL["builtins.quit"] = builtins.quit
    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    for name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
        "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
        "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
        "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    ):
        if hasattr(os, name):
            setattr(os, name, None)

    shutil.rmtree = None
    shutil.move = None
    if hasattr(shutil, "chown"):
        shutil.chown = None

    import subprocess as _sp
    _sp.Popen = None  # type: ignore[attr-defined]

    try:
        __builtins__["help"] = None  # type: ignore[index]
    except (TypeError, KeyError):
        pass

    for mod in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[mod] = None  # type: ignore[assignment]
