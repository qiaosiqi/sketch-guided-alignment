"""
模型输出解析。

prompt 结尾会主动以 `<SKETCH>\n` 或 ```` ```python\n ```` 结束(见 data/prompts.py),
所以 completion 文本本身就是 sketch / code 主体的开头。stop 序列可能已经截掉了结尾。
保险起见这里再做一遍正则清理。

Sketch:
    优先匹配第一个 </SKETCH> 之前的全部内容
    没有 </SKETCH> 时,截到第一个 "\n### " 或文本末尾
    最终去除首尾空白,如果是空字符串视为解析失败

Code:
    优先匹配第一个 ``` 之前的全部内容(prompt 已经给了 ```python\n)
    没有 ``` 时,截到 "\n### " 或文本末尾
    去除首尾空白和孤立的 ```python 行
    空字符串视为解析失败
"""
from __future__ import annotations
import re


_END_HEADER = re.compile(r"\n###\s")


def parse_sketch(completion: str) -> tuple[str, bool]:
    """返回 (sketch_text, ok)。"""
    text = completion

    # 截到 </SKETCH>
    idx = text.find("</SKETCH>")
    if idx != -1:
        text = text[:idx]
    else:
        # 截到下一个 ### 段标题
        m = _END_HEADER.search(text)
        if m:
            text = text[:m.start()]

    text = text.strip()
    # 移除残留的 <SKETCH> 起始标签(罕见,但模型偶尔吐)
    text = re.sub(r"^\s*<SKETCH>\s*", "", text)
    if not text:
        return "", False
    # 太短的 sketch (<10 字符) 也视为失败
    if len(text) < 10:
        return text, False
    return text, True


_FENCE = re.compile(r"```")
_LEADING_PY = re.compile(r"^\s*```(?:python)?\s*\n")


def parse_code(completion: str) -> tuple[str, bool]:
    """返回 (code_text, ok)。"""
    text = completion

    # 去掉模型可能复述的开头 ```python
    text = _LEADING_PY.sub("", text)

    # 截到下一个 ```
    m = _FENCE.search(text)
    if m:
        text = text[:m.start()]
    else:
        # 截到下一个 ### 段
        m = _END_HEADER.search(text)
        if m:
            text = text[:m.start()]

    text = text.rstrip()
    # 去掉行首 / 行尾空行
    while text.startswith("\n"):
        text = text[1:]
    if not text:
        return "", False
    # 至少包含一个真正的代码行:有 ":" 或 "=" 或 "import" 或 "def"
    if not re.search(r"(:\s*$|=|\bimport\b|\bdef\b|\bclass\b|\bprint\b)", text, re.MULTILINE):
        return text, False
    return text, True
