"""
測試重點：
- CLOSED 狀態下正常呼叫成功。
- 連續失敗達 threshold 後轉為 OPEN，並拒絕後續呼叫。
- OPEN 狀態經過 recovery_timeout 後轉 HALF_OPEN。
- HALF_OPEN 連續成功達 success_threshold 後轉回 CLOSED。
- HALF_OPEN 中只要失敗一次就退回 OPEN。
- Registry 會依名稱建立獨立的 breaker，互不影響。
"""

import asyncio
import pytest

from circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
)


async def _ok_func():
    return "ok"


async def _fail_func():
    raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_closed_state_success_passthrough():
    cb = CircuitBreaker(name="test", config=CircuitBreakerConfig(failure_threshold=3))
    result = await cb.call(_ok_func)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_opens_after_threshold_failures():
    cb = CircuitBreaker(name="test", config=CircuitBreakerConfig(failure_threshold=3))
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(_fail_func)
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_state_rejects_calls_immediately():
    cb = CircuitBreaker(
        name="test",
        config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout=999),
    )
    with pytest.raises(RuntimeError):
        await cb.call(_fail_func)
    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(_ok_func)


@pytest.mark.asyncio
async def test_transitions_to_half_open_after_recovery_timeout():
    cb = CircuitBreaker(
        name="test",
        config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05),
    )
    with pytest.raises(RuntimeError):
        await cb.call(_fail_func)
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.06)
    assert cb.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_half_open_recovers_to_closed_after_success_threshold():
    cb = CircuitBreaker(
        name="test",
        config=CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.05,
            half_open_max_calls=5,
            success_threshold=2,
        ),
    )
    with pytest.raises(RuntimeError):
        await cb.call(_fail_func)
    await asyncio.sleep(0.06)
    assert cb.state == CircuitState.HALF_OPEN

    await cb.call(_ok_func)
    # 還沒達到 success_threshold=2，應該還在 HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN

    await cb.call(_ok_func)
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_failure_reopens_circuit():
    cb = CircuitBreaker(
        name="test",
        config=CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.05, half_open_max_calls=5
        ),
    )
    with pytest.raises(RuntimeError):
        await cb.call(_fail_func)
    await asyncio.sleep(0.06)
    assert cb.state == CircuitState.HALF_OPEN

    with pytest.raises(RuntimeError):
        await cb.call(_fail_func)
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_half_open_respects_max_calls_limit():
    cb = CircuitBreaker(
        name="test",
        config=CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.05, half_open_max_calls=1
        ),
    )
    with pytest.raises(RuntimeError):
        await cb.call(_fail_func)
    await asyncio.sleep(0.06)
    assert cb.state == CircuitState.HALF_OPEN

    async def _slow_ok():
        await asyncio.sleep(0.02)
        return "ok"

    task = asyncio.create_task(cb.call(_slow_ok))
    await asyncio.sleep(0.005)  # 讓第一個呼叫先佔用 half_open_calls 計數
    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(_ok_func)
    await task


@pytest.mark.asyncio
async def test_registry_creates_independent_breakers():
    registry = CircuitBreakerRegistry(
        default_config=CircuitBreakerConfig(failure_threshold=1)
    )
    cb_a = registry.get("tool_a")
    cb_b = registry.get("tool_b")
    assert cb_a is not cb_b

    with pytest.raises(RuntimeError):
        await cb_a.call(_fail_func)
    assert cb_a.state == CircuitState.OPEN
    assert cb_b.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_registry_get_is_idempotent_by_name():
    registry = CircuitBreakerRegistry()
    cb1 = registry.get("same_tool")
    cb2 = registry.get("same_tool")
    assert cb1 is cb2


def test_status_report_reflects_states():
    registry = CircuitBreakerRegistry(
        default_config=CircuitBreakerConfig(failure_threshold=1)
    )
    registry.get("tool_a")
    report = registry.status_report()
    assert report == {"tool_a": "closed"}
