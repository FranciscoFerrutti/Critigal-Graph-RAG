"""
Builder del StateGraph LangGraph.

Patrón ReAct (Reasoning + Acting):
  process → [tool_calls?] → tools → process → finalize → END

Sin tools, se va directo de process a finalize.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes import AgentNodes
from src.agent.state import AgentState


def build_graph(nodes: AgentNodes, tools: list[StructuredTool]) -> StateGraph:
    """
    Arma el StateGraph con los nodos y conexiones del agente.

    Args:
        nodes: Instancia de AgentNodes con LLM + tools inyectadas.
        tools: Lista de tools (para construir el ToolNode). Si está vacía,
               el grafo salta el branch de tools.
    """
    graph = StateGraph(AgentState)

    graph.add_node("process", nodes.process_message)
    graph.add_node("finalize", nodes.finalize_response)

    graph.add_edge(START, "process")

    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_conditional_edges(
            "process",
            nodes.should_use_tools,
            {"tools": "tools", "finalize": "finalize"},
        )
        graph.add_edge("tools", "process")
    else:
        graph.add_edge("process", "finalize")

    graph.add_edge("finalize", END)
    return graph
