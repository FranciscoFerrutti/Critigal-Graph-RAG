"""
Orquestador principal: ChatbotAgent + singleton helpers.

Wire-up de configs, LLM, tools (via registry) y grafo LangGraph.
"""

from __future__ import annotations

import ast
import json
import logging
import time
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

# Mapeo de cada tool a la(s) fuente(s) de datos que consulta. Hoy todas las
# tools leen del Knowledge Graph derivado de ACLED, por lo que todas mapean a
# "ACLED". Cuando se integren proveedores nuevos (otra base, otra API), basta
# extender este dict —o moverlo a config— sin tocar el server ni el proxy.
_TOOL_DATA_SOURCES: dict[str, list[str]] = {
    "similarity_search": ["ACLED"],
    "cypher_query": ["ACLED"],
    "text2cypher": ["ACLED"],
}
# Fuente asumida para una tool sin mapeo explícito (failsafe: la única que hay).
_DEFAULT_DATA_SOURCE = "ACLED"

# Tope de citas devueltas y largo máximo de cada `notes` (evita payloads enormes
# cuando una búsqueda vectorial trae muchos eventos con notas largas).
_MAX_CITATIONS = 20
_NOTES_MAX_CHARS = 500


def _parse_tool_content(content: Any) -> list[dict[str, Any]]:
    """
    Recupera la(s) fila(s) crudas de un `ToolMessage`.

    Las tools devuelven `list[dict]`, pero LangChain serializa el content del
    `ToolMessage` a string (JSON, o repr de Python como fallback). Esta función
    revierte esa serialización a una lista de dicts. Si no se puede parsear,
    devuelve `[]` (degradación silenciosa: sin citas, no error).
    """
    if isinstance(content, list):
        return [r for r in content if isinstance(r, dict)]
    if isinstance(content, dict):
        return [content]
    if not isinstance(content, str) or not content.strip():
        return []
    s = content.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(s)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        return []
    return []


def _extract_citations(messages: list[Any]) -> list[dict[str, Any]]:
    """
    Extrae las citas (eventos ACLED concretos) que fundamentan la respuesta.

    Recorre los `ToolMessage` del state y toma cada fila que tenga `event_id`
    (las que vienen de `similarity_search`, o de cualquier query que devuelva
    eventos). Deduplica por `event_id`, trunca `notes` y anota la fuente de
    datos según la tool que produjo el resultado.
    """
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for msg in messages:
        if msg.__class__.__name__ != "ToolMessage":
            continue
        tool_name = getattr(msg, "name", "") or ""
        sources = _TOOL_DATA_SOURCES.get(tool_name, [_DEFAULT_DATA_SOURCE])
        source = sources[0] if sources else _DEFAULT_DATA_SOURCE
        for row in _parse_tool_content(getattr(msg, "content", None)):
            event_id = row.get("event_id")
            if not event_id or str(event_id) in seen:
                continue
            seen.add(str(event_id))
            notes = row.get("notes")
            if isinstance(notes, str) and len(notes) > _NOTES_MAX_CHARS:
                notes = notes[:_NOTES_MAX_CHARS].rstrip() + "…"
            citations.append({
                "event_id": str(event_id),
                "event_date": row.get("event_date"),
                "country": row.get("country"),
                "admin1": row.get("admin1"),
                "notes": notes,
                "score": row.get("score"),
                "source": source,
            })
            if len(citations) >= _MAX_CITATIONS:
                return citations
    return citations


class ChatbotAgent:
    """Agente conversacional Critical Graph RAG."""

    def __init__(self, config_dir: str = "config") -> None:
        self.config_dir = config_dir

        # Configs
        self.agent_config = load_yaml(f"{config_dir}/agent.yaml")
        self.embeddings_config = load_yaml(f"{config_dir}/embeddings.yaml")
        self.graph_schema = GraphSchema.from_yaml(f"{config_dir}/graph_schema.yaml")

        # LLM
        provider = self.agent_config.get("provider", "google")
        api_key = resolve_api_key(self.agent_config[provider]["api_key_env"])
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
            dict con `response`, `tool_calls`, `used_tool`, `used_tools`,
            `data_sources`, `citations`, `latency_ms` y `messages_count`.
        """
        initial_state = AgentState(messages=[HumanMessage(content=user_message)])
        _start = time.perf_counter()
        final_state = self.runnable.invoke(initial_state)
        latency_ms = int((time.perf_counter() - _start) * 1000)

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

        # Tools efectivamente usadas (nombres distintos, en orden de llamada).
        used_tools: list[str] = []
        for tc in tool_calls:
            name = tc.get("name") or ""
            if name and name not in used_tools:
                used_tools.append(name)

        # Fuentes de datos derivadas de las tools usadas (unión sin duplicados).
        data_sources: list[str] = []
        for name in used_tools:
            for source in _TOOL_DATA_SOURCES.get(name, [_DEFAULT_DATA_SOURCE]):
                if source not in data_sources:
                    data_sources.append(source)

        citations = _extract_citations(final_state.get("messages", []))

        return {
            "response": response,
            "tool_calls": tool_calls,
            # `used_tool`: tool única que resolvió la pregunta (caso habitual).
            # Es None si el agente respondió sin tools. Cuando se usa más de una
            # tool en el mismo turno, ver `used_tools` para la lista completa.
            "used_tool": used_tools[0] if used_tools else None,
            "used_tools": used_tools,
            "data_sources": data_sources,
            "citations": citations,
            "latency_ms": latency_ms,
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
