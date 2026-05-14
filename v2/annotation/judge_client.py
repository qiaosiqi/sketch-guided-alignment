"""
GLM-4-Air HTTP 客户端。

ZhipuAI v4 API endpoint:
    https://open.bigmodel.cn/api/paas/v4/chat/completions

认证:
    Authorization: Bearer {API_KEY}
    API key 从环境变量 GLM_API_KEY 读取。

注意:ZhipuAI 历史上有 JWT 认证方式(id.secret 拆分签 token),v4 起允许直接用 API key
作为 Bearer。若你的 key 是 JWT 版,在 build_judge() 里传 jwt_token 即可。

调用:
    judge = GLMJudge(model="glm-4-air", temperature=0.0, max_tokens=1024)
    text = judge.chat(system_prompt, user_prompt)

设计要点:
- 3 次重试,指数退避(1s, 2s, 4s)
- 任何 5xx / 网络异常都重试;4xx 不重试(配置/认证错误)
- 默认 temperature=0(deterministic),max_tokens=1024
- 不依赖 openai/zhipuai SDK,只用 requests
"""
from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


class JudgeError(RuntimeError):
    """非重试型错误(4xx / 参数错误)。"""


@dataclass
class GLMJudge:
    api_key: Optional[str] = None
    endpoint: str = DEFAULT_ENDPOINT
    model: str = "glm-4-air"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 3

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("GLM_API_KEY")
        if not self.api_key:
            raise JudgeError("GLM_API_KEY not set (env var) and api_key not passed.")
        self._session = requests.Session()

    # ---------- 主调用 ----------

    def chat(self, system: str, user: str) -> str:
        """单轮对话。返回 assistant 的 text 部分。失败抛 JudgeError。"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        delay = 1.0
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.post(
                    self.endpoint, json=payload, headers=headers, timeout=self.timeout,
                )
                if r.status_code == 200:
                    data = r.json()
                    return data["choices"][0]["message"]["content"]
                # 4xx:不重试
                if 400 <= r.status_code < 500:
                    raise JudgeError(f"HTTP {r.status_code}: {r.text[:300]}")
                # 5xx:重试
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                last_err = e
            log.warning(f"judge call failed (attempt {attempt}/{self.max_retries}): {last_err}")
            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2
        raise JudgeError(f"max retries exhausted; last error: {last_err}")


def build_judge(**kwargs) -> GLMJudge:
    """工厂函数。预留扩展位(后期可换 OpenAI/Claude)。"""
    return GLMJudge(**kwargs)
