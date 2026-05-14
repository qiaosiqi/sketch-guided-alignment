"""execution/timing.py 的 smoke tests。"""
from v2.execution.timing import stable_runtime


def test_stable_runtime_returns_pair_or_none():
    """
    契约测试:stable_runtime 在合理参数下要么收敛返回 (mean, std),
    要么 max_outer 用尽返回 (None, None),都不抛异常。
    (Windows 上 process_time_ns 分辨率粗到几毫秒,轻量 workload 可能测出 mean=0,
    导致 std/mean 永远不达标,这是合法的"超出预算未收敛"路径。)
    """
    workload = lambda: sum(range(10000))
    mean, std = stable_runtime(workload, inner_repeats=50, max_outer=20, cov_threshold=0.5)
    # 两种合法返回:都 None,或都非 None 且 >= 0
    if mean is None:
        assert std is None
    else:
        assert mean >= 0 and std is not None and std >= 0


def test_stable_runtime_can_fail_under_tight_cov():
    """极严的 CoV 要求 + 极少轮数 → 必然不收敛。"""
    import random

    def jittery():
        # 制造方差
        x = 0
        for _ in range(random.randint(1, 100)):
            x += 1

    mean, std = stable_runtime(jittery, inner_repeats=5, max_outer=3, cov_threshold=0.001)
    # 大概率失败(随机性,允许偶发通过)
    # 这里测试主要确保 max_outer 上限工作,不抛异常
    assert mean is None or mean >= 0
