"""
Orquestador principal: ChatbotAgent + singleton helpers.

Wire-up de configs, LLM, tools (via registry) y grafo LangGraph.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.agent.graph import build_graph
from src.agent.lang import detect_language
from src.agent.llm import build_planner_llm
from src.agent.nodes import AgentNodes
from src.agent.state import AgentState
from src.config import load_yaml, resolve_api_key
from src.schema import GraphSchema
from src.tools import ToolContext, available_tools, build_tools

logger = logging.getLogger(__name__)


class ChatbotAgent:
    """Agente conversacional Critical Graph RAG."""

    def __init__(self, config_dir: str = "config") -> None:
        self.config_dir = config_dir

        # Configs
        self.agent_config = load_yaml(f"{config_dir}/agent.yaml")
        self.embeddings_config = load_yaml(f"{config_dir}/embeddings.yaml")
        self.graph_schema = GraphSchema.from_yaml(f"{config_dir}/graph_schema.yaml")

        # LLM
        api_key = resolve_api_key(self.agent_config["google"]["api_key_env"])
        self.planner_llm = build_planner_llm(self.agent_config, api_key)

        # Tools (vía registry; enabled list opcional en agent.yaml)
        ctx = ToolContext(
            agent_config=self.agent_config,
            embeddings_config=self.embeddings_config,
            graph_schema=self.graph_schema,
        )
        enabled = self.agent_config.get("enabled_tools")
        try:
            self.tools = build_tools(ctx, enabled=enabled)
        except Exception:
            logger.exception(
                "Fallo construyendo tools (enabled=%s, registradas=%s)",
                enabled, available_tools(),
            )
            raise

        # Grafo
        self.nodes = AgentNodes(self.planner_llm, self.tools)
        self.graph = build_graph(self.nodes, self.tools)
        self.runnable = self.graph.compile()

        logger.info(
            "✓ Agente inicializado con %d tool(s): %s",
            len(self.tools),
            [t.name for t in self.tools],
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def invoke(self, user_message: str) -> str:
        """Procesa una pregunta y devuelve la respuesta del agente."""
        try:
            return self.invoke_trace(user_message)["response"]
        except Exception as e:
            logger.error("Error en agente: %s", e, exc_info=True)
            language = detect_language(user_message)
            if language == "es":
                return (
                    "Perdón, en este momento no puedo atender su consulta. "
                    "Por favor, intente más tarde."
                )
            return (
                "Sorry, I can't process your request right now. "
                "Please try again later."
            )

    def invoke_trace(self, user_message: str) -> dict[str, Any]:
        """
        Variante de `invoke` que devuelve también la traza de tools llamadas.

        Returns:
            dict con `response`, `tool_calls` y `messages_count`.
        """
        initial_state = AgentState(messages=[HumanMessage(content=user_message)])
        final_state = self.runnable.invoke(initial_state)

        response = (
            final_state.get("response")
            or final_state.get("error", "")
            or "No se pudo procesar tu pregunta."
        )

        tool_calls: list[dict[str, Any]] = []
        for msg in final_state.get("messages", []):
            for tc in getattr(msg, "tool_calls", None) or []:
                if isinstance(tc, dict):
                    tool_calls.append({"name": tc.get("name", ""), "args": tc.get("args", {})})
                else:
                    tool_calls.append(
                        {"name": getattr(tc, "name", ""), "args": getattr(tc, "args", {})}
                    )

        return {
            "response": response,
            "tool_calls": tool_calls,
            "messages_count": len(final_state.get("messages", [])),
        }


# ---------------------------------------------------------------------------
# Singleton (lifecycle FastAPI)
# ---------------------------------------------------------------------------

_agent_instance: ChatbotAgent | None = None


def get_agent() -> ChatbotAgent:
    """Devuelve la instancia singleton del agente (lazy init si hace falta)."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ChatbotAgent()
    return _agent_instance


def init_agent(config_dir: str = "config") -> ChatbotAgent:
    """Inicializa el agente explícitamente. Llamar desde startup."""
    global _agent_instance
    _agent_instance = ChatbotAgent(config_dir=config_dir)
    return _agent_instance
