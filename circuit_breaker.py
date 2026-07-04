"""三態斷路器（CLOSED / OPEN / HALF_OPEN）與全域 Registry。"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("hermes.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """電路開啟時拒絕請求"""
    pass


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._recovery_timeout_elapsed():
            logger.info(f"breaker[{self.name}] 從 OPEN 轉為 HALF_OPEN")
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._success_count = 0
        return self._state

    def _recovery_timeout_elapsed(self) -> bool:
        return (time.monotonic() - self._last_failure_time) >= self.config.recovery_timeout

    async def call(self, func: Callable[[], Any], *args, **kwargs) -> Any:
        current_state = self.state
        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"breaker[{self.name}] OPEN，拒絕請求")

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpenError(f"breaker[{self.name}] HALF_OPEN（試探中），已達上限")

            self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                logger.info(f"breaker[{self.name}] 從 HALF_OPEN 恢復為 CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                self._success_count = 0
        else:
            self._failure_count = 0

    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"breaker[{self.name}] HALF_OPEN 中失敗，退回 OPEN")
            self._state = CircuitState.OPEN
            self._half_open_calls = 0
            self._success_count = 0
        elif self._failure_count >= self.config.failure_threshold:
            logger.warning(f"breaker[{self.name}] 連續失敗 {self._failure_count} 次，轉為 OPEN")
            self._state = CircuitState.OPEN
            self._half_open_calls = 0
            self._success_count = 0


class CircuitBreakerRegistry:
    def __init__(self, default_config: Optional[CircuitBreakerConfig] = None):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = default_config or CircuitBreakerConfig()

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config=self._default_config)
        return self._breakers[name]

    def status_report(self) -> dict[str, str]:
        return {name: cb.state.value for name, cb in self._breakers.items()}


mcp_circuit_registry = CircuitBreakerRegistry()
