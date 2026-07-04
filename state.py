"""LangGraph 狀態定義。"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class HermesState(TypedDict, total=False):
    user_input: str
    messages: List[Any]
    completed_agents: List[str]
    agent_outputs: Dict[str, Any]
    errors: List[str]
    step_count: int
    max_steps: int
    final_answer: str
    next_agent: str
