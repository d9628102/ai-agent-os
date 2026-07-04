"""共用 pytest fixture 與 sys.path 設定。"""

import sys
import os
import pytest
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def reset_circuit_registry():
    """每個測試前重置全域 circuit breaker registry，避免測試互相污染狀態。"""
    from circuit_breaker import mcp_circuit_registry
    mcp_circuit_registry._breakers.clear()
    yield
    mcp_circuit_registry._breakers.clear()


@pytest.fixture(scope="session")
def event_loop():
    """讓 async fixtures 可以在 session scope 正常運作（Python 3.9 compatibility）。"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
