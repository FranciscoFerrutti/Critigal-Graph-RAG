"""Estado compartido del agente conversacional."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict


class AgentState(BaseModel):
    """
    Estado del grafo LangGraph.

    `messages` usa el reducer `add_messages` para que cada nodo devuelva
    solo el delta (no toda la lista). Sin esto, `ToolNode` rompe el historial.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: dict[str, Any] | None = None
    response: str | None = None
    error: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
