"""
Nodos del grafo LangGraph.

Se agrupan en `AgentNodes` para que los nodos compartan `planner_llm` y
`tools` sin globals ni cierre implícito. Cada método es un nodo enchufable
en el StateGraph.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from src.agent.lang import detect_language
from src.agent.prompts import get_system_prompt
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


class AgentNodes:
    """Conjunto de nodos del grafo, parametrizado con LLM + tools."""

    def __init__(self, planner_llm: BaseChatModel, tools: list[StructuredTool]) -> None:
        self.planner_llm = planner_llm
        self.tools = tools

    # ------------------------------------------------------------------
    # Edge condicional
    # ------------------------------------------------------------------

    def should_use_tools(self, state: AgentState) -> str:
        """Si el último AIMessage trae tool_calls → 'tools', si no → 'finalize'."""
        last_msg = state.messages[-1]
        if getattr(last_msg, "tool_calls", None):
            return "tools"
        return "finalize"

    # ------------------------------------------------------------------
    # Nodos
    # ------------------------------------------------------------------

    def process_message(self, state: AgentState) -> dict[str, Any]:
        """
        Dos modos:
          - 1ra iter (sin ToolMessages): planner con tools bound, decide qué tool usar.
          - 2da iter (con ToolMessages): NO reenvía historial — gemini-2.5-flash
            devuelve AIMessage con content="" + tool_calls cuyo segundo envío rompe
            con `contents are required`. Se arma prompt nuevo con la pregunta
            original + el resultado crudo y se pide síntesis sin tools.
        """
        user_text = _first_human_text(state)
        language = detect_language(user_text)
        system_prompt = get_system_prompt(language)

        tool_msgs = [m for m in state.messages if m.__class__.__name__ == "ToolMessage"]
        tool_calls_seen = _collect_tool_calls(state)

        if not tool_msgs:
            llm = self.planner_llm.bind_tools(self.tools) if self.tools else self.planner_llm
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]
            response = llm.invoke(messages)
            if getattr(response, "tool_calls", None):
                for tc in response.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    logger.info("Tool seleccionada: %s (args=%s)", name, args)
        else:
            tool_context = _format_tool_context(tool_calls_seen, tool_msgs)
            synth_user = (
                f"Pregunta original del usuario:\n{user_text}\n\n"
                f"Resultado(s) de la(s) tool(s):\n{tool_context}\n\n"
                f"Generá una respuesta concisa y clara basada SOLO en estos resultados. "
                f"Si el resultado es numérico, expresalo con la unidad implícita. "
                f"Si es una lista, formateala con bullets. "
                f"Respondé en el mismo idioma que la pregunta."
            )
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=synth_user)]
            response = self.planner_llm.invoke(messages)

        return {"messages": [response]}

    def finalize_response(self, state: AgentState) -> dict[str, Any]:
        """Extrae el último AIMessage con texto y lo expone en `state.response`."""
        for msg in reversed(state.messages):
            if msg.__class__.__name__ == "AIMessage":
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content:
                    return {"response": content}

        last_msg = state.messages[-1]
        content = getattr(last_msg, "content", "")
        text = content if isinstance(content, str) else str(content)
        return {"response": text or "No se pudo generar una respuesta."}


# ---------------------------------------------------------------------------
# Helpers de inspección de state
# ---------------------------------------------------------------------------

def _first_human_text(state: AgentState) -> str:
    for m in state.messages:
        if m.__class__.__name__ == "HumanMessage":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _collect_tool_calls(state: AgentState) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in state.messages:
        calls = getattr(m, "tool_calls", None) or []
        for tc in calls:
            if isinstance(tc, dict):
                out.append(tc)
            else:
                out.append(
                    {"name": getattr(tc, "name", ""), "args": getattr(tc, "args", {})}
                )
    return out


def _format_tool_context(tool_calls: list[dict[str, Any]], tool_msgs: list[Any]) -> str:
    if not tool_msgs:
        return "(sin resultados de tool)"
    lines = []
    for tc, tm in zip(tool_calls, tool_msgs):
        lines.append(
            f"Tool `{tc.get('name', '?')}` llamada con args={tc.get('args', {})}\n"
            f"Resultado: {tm.content}"
        )
    return "\n\n".join(lines)
