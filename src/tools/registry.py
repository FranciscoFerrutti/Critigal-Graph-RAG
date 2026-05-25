"""
Tool registry + ToolContext.

Patrón plugin: cada tool se auto-registra con `@register_tool("name")` y
recibe un `ToolContext` con todas las dependencias compartidas
(configs, schema, futuros: driver, cache, telemetry).

Agregar una tool nueva = un file con un builder decorado, sin tocar el
agente. Habilitar/deshabilitar desde `agent.yaml` (`enabled_tools: [...]`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from src.schema import GraphSchema


@dataclass(frozen=True)
class ToolContext:
    """
    Contenedor de dependencias inyectadas a los builders de tools.

    Mantener `frozen=True` para que los builders no muten el contexto.
    Extender agregando campos opcionales con default — no rompe builders existentes.
    """

    agent_config: dict[str, Any]
    embeddings_config: dict[str, Any]
    graph_schema: GraphSchema


ToolBuilder = Callable[[ToolContext], StructuredTool]

_REGISTRY: dict[str, ToolBuilder] = {}


def register_tool(name: str) -> Callable[[ToolBuilder], ToolBuilder]:
    """
    Decorador para registrar un builder de tool bajo `name`.

    El builder debe aceptar un único arg `ctx: ToolContext` y devolver
    una `StructuredTool` lista para enchufar al agente.

    Raises:
        ValueError: si `name` ya está registrado (evita shadowing silencioso).
    """

    def deco(fn: ToolBuilder) -> ToolBuilder:
        if name in _REGISTRY:
            raise ValueError(f"Tool '{name}' ya registrada por otro builder.")
        _REGISTRY[name] = fn
        return fn

    return deco


def available_tools() -> list[str]:
    """Lista de nombres de tools registradas."""
    return sorted(_REGISTRY)


def build_tools(ctx: ToolContext, enabled: list[str] | None = None) -> list[StructuredTool]:
    """
    Instancia las tools habilitadas con el contexto dado.

    Args:
        ctx: Dependencias compartidas.
        enabled: Lista de nombres a instanciar. Si es None, instancia todas
                 las registradas. Nombres desconocidos lanzan ValueError.

    Returns:
        Lista de StructuredTool en el orden de `enabled` (o registro alfabético).
    """
    names = enabled if enabled is not None else available_tools()
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise ValueError(
            f"Tools desconocidas en enabled_tools: {unknown}. "
            f"Registradas: {available_tools()}"
        )
    return [_REGISTRY[n](ctx) for n in names]


def _reset_registry_for_tests() -> None:
    """Solo para tests: limpia el registro global."""
    _REGISTRY.clear()
