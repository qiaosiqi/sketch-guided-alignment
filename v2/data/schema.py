"""
数据契约。所有阶段间传递的对象都用这里的 dataclass / TypedDict 定义。
JSON 序列化用 dataclasses.asdict()。
"""
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


IOFormat = Literal["stdio", "fncall"]


@dataclass
class Problem:
    """一道 APPS 题。apps_loader 产出,后续所有阶段消费。"""
    task_id: str                          # 例 "apps_train_1266"
    difficulty: Literal["introductory", "interview", "competition"]
    io_format: IOFormat
    question: str                         # 原始题面
    inputs: list                          # 测试用例输入:list of test cases
    outputs: list                         # 期望输出:list of test cases
    fn_name: Optional[str] = None         # fncall 时函数名,stdio 为 None
    starter_code: Optional[str] = None    # 部分题给的起始代码片段
    url: Optional[str] = None             # metadata.json 里的来源链接


@dataclass
class SketchSample:
    """Stage-1 采样产物。"""
    task_id: str
    sample_id: int                        # 0..N-1
    sketch: str                           # 模型生成的算法 sketch 文本
    sample_temp: float
    raw_completion: str                   # 原始未截断的输出,debug 用
    parsed_ok: bool                       # 解析是否成功


@dataclass
class CodeSample:
    """Stage-2 采样产物。每个 SketchSample 配一个 CodeSample(或多个,看配置)。"""
    task_id: str
    sample_id: int                        # 沿用 sketch 的 sample_id
    code_id: int = 0                      # 同一 sketch 下第几个 code(默认 1 个就只 0)
    sketch: str = ""                      # 冗余存一份,方便单独消费
    code: str = ""
    sample_temp: float = 0.6
    raw_completion: str = ""
    parsed_ok: bool = True


@dataclass
class ExecutionResult:
    """execution/runner 产出。每个 CodeSample 一条。"""
    task_id: str
    sample_id: int
    code_id: int
    n_tests: int
    n_passed: int
    pass_ratio: float                     # n_passed / n_tests
    per_test_pass: list                   # [bool, ...] 长度 n_tests
    error: Optional[str] = None           # 若整体抛异常(语法错/超时),记原因
    runtime_ns_mean: Optional[float] = None   # 仅当 pass_ratio == 1.0 时测,否则 None
    runtime_ns_std: Optional[float] = None
    timed_out: bool = False


@dataclass
class SketchScores:
    """Sketch 维度 0-10 分。"""
    S1_correctness: float                 # 算法正确性
    S2_specificity: float                 # 具体性(不能空话)
    S3_complexity_awareness: float        # 复杂度意识
    S4_edge_coverage: float               # 边界覆盖
    raw_response: str = ""                # judge 原始回复,debug 用


@dataclass
class CodeScores:
    """Code 维度 0-10 分。"""
    C1_faithfulness: float                # 代码忠于 sketch 的程度(本方法核心维度)
    C2_time_complexity: float
    C3_space_complexity: float
    C4_readability: float
    C5_edge_handling: float
    raw_response: str = ""


@dataclass
class AlgorithmScores:
    """完整评分。"""
    sketch: SketchScores
    code: CodeScores
    final: float                          # 0.4 * mean(S) + 0.6 * mean(C)
    judge_model: str = "GLM-4-Air"
    parsable: bool = True                 # 任一阶段未能解析则 False,final 设为 -1


# ---------- merged.jsonl 的最终条目 schema ----------

@dataclass
class MergedAnswer:
    """merged.jsonl 中 answers 数组的一项。"""
    sketch: str
    code: str
    pass_ratio: float
    n_tests: int
    runtime_ns_mean: Optional[float] = None
    runtime_ns_std: Optional[float] = None
    algo_final: float = -1.0
    algo_breakdown: Optional[dict] = None     # asdict(AlgorithmScores) 但只留分数
    sample_temp: float = 0.6


@dataclass
class MergedProblem:
    """merged.jsonl 中每行一题。"""
    task_id: str
    difficulty: str
    io_format: str
    question: str
    fn_name: Optional[str] = None
    answers: list = field(default_factory=list)   # List[MergedAnswer]


# ---------- 序列化辅助 ----------

def to_dict(obj):
    return asdict(obj)
