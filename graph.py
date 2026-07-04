"""
graph.py
強化模組整合：把 Orchestrator / Research / Document / Finance / Synthesizer
組裝成 LangGraph StateGraph，並用 AsyncPostgresSaver 做狀態持久化。
"""

import logging
from langgraph.graph import StateGraph, END

from state import HermesState

logger = logging.getLogger("hermes.graph")


def build_graph_builder() -> StateGraph:
    """
    純粹組裝節點與邊（不含 checkpointer），讓測試可以用 MemorySaver
    取代真正的 Postgres，驗證路由邏輯本身是否正確。
    正式環境請用 compile_graph()，會接上 AsyncPostgresSaver。
    """
    from agents.orchestrator import orchestrator_node, route_after_orchestrator
    from agents.research import research_node
    from agents.document import document_node
    from agents.finance import finance_node
    from agents.synthesizer import synthesizer_node

    builder = StateGraph(HermesState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("research", research_node)
    builder.add_node("document", document_node)
    builder.add_node("finance", finance_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.set_entry_point("orchestrator")

    # 條件路由：orchestrator 判斷完後分派到對應 agent
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "research": "research",
            "document": "document",
            "finance": "finance",
            "synthesizer": "synthesizer",
        },
    )

    # 各 agent 執行完後都回到 orchestrator，由它決定下一步（形成迴圈直到轉 synthesizer）
    builder.add_edge("research", "orchestrator")
    builder.add_edge("document", "orchestrator")
    builder.add_edge("finance", "orchestrator")

    # synthesizer 完成後結束流程
    builder.add_edge("synthesizer", END)

    return builder


async def compile_graph(checkpointer=None):
    """
    編譯 LangGraph。若傳入 checkpointer（MemorySaver / AsyncPostgresSaver）則啟用狀態持久化。
    """
    builder = build_graph_builder()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


async def build_hermes_graph(postgres_conn_string: str):
    """
    建立並編譯 Hermes Agent 的 LangGraph 圖，使用 AsyncPostgresSaver 做狀態持久化。

    postgres_conn_string 範例：
        "postgresql://user:password@localhost:5432/hermes_db"
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    builder = build_graph_builder()

    async with AsyncPostgresSaver.from_conn_string(postgres_conn_string) as checkpointer:
        await checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
        return graph, checkpointer


async def run_hermes_query(
    graph,
    user_input: str,
    thread_id: str,
    max_steps: int = 8,
) -> dict:
    """
    執行單次查詢，thread_id 用於 Postgres checkpoint 區分不同對話/任務。
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: HermesState = {
        "user_input": user_input,
        "messages": [],
        "completed_agents": [],
        "agent_outputs": {},
        "errors": [],
        "step_count": 0,
        "max_steps": max_steps,
    }

    final_state = await graph.ainvoke(initial_state, config=config)
    return {
        "final_answer": final_state.get("final_answer", ""),
        "agent_outputs": final_state.get("agent_outputs", {}),
        "errors": final_state.get("errors", []),
        "completed_agents": final_state.get("completed_agents", []),
    }
