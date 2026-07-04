"""Exponential Backoff Retry 包裝 LangChain LLM invoke。"""

import asyncio
import logging
import random
from typing import Any, List

from langchain_core.messages import BaseMessage

logger = logging.getLogger("hermes.llm_retry")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(keyword in msg for keyword in ["429", "503", "overloaded", "rate limit", "service unavailable"])


async def llm_ainvoke(
    llm: Any,
    messages: List[BaseMessage],
    max_attempts: int = 4,
    wait_min: float = 1.0,
    wait_max: float = 30.0,
) -> Any:
    """包裝 LLM.ainvoke，遇上 retryable 錯誤時 exponential backoff + jitter。"""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await llm.ainvoke(messages)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
            if attempt == max_attempts:
                raise

            delay = min(wait_min * (2 ** (attempt - 1)) + random.uniform(0, 0.5), wait_max)
            logger.warning(
                f"LLM 呼叫失敗（第 {attempt} 次），{delay:.1f}s 後重試: {exc}"
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore
