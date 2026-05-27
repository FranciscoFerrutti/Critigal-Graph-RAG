"""
Tools expuestas al agente.

Tres herramientas:
  - similarity_search : búsqueda vectorial sobre Event.notes (funcional).
  - cypher_query      : queries predefinidas de config/cypher_library.yaml.
  - text2cypher       : fallback — LLM traduce pregunta -> Cypher (scaffolding).

Cada tool se auto-registra en `src.tools.registry` vía decorator. Importar
los submódulos acá garantiza que el registry esté poblado antes de que
el agente lo consulte.
"""

from src.tools import cypher_query as _cypher_query  # noqa: F401  (registra tool)
from src.tools import similarity_search as _similarity_search  # noqa: F401  (registra tool)
from src.tools import text2cypher as _text2cypher  # noqa: F401  (registra tool)
from src.tools.registry import ToolContext, available_tools, build_tools, register_tool

__all__ = [
    "ToolContext",
    "available_tools",
    "build_tools",
    "register_tool",
]


def __getattr__(name: str):
    """Lazy access para APIs legacy (mantener compat con código antiguo)."""
    if name == "build_similarity_search_tool":
        from src.tools.similarity_search import build_similarity_search_tool
        return build_similarity_search_tool
    if name == "build_cypher_query_tool":
        from src.tools.cypher_query import build_cypher_query_tool
        return build_cypher_query_tool
    if name == "CypherLibrary":
        from src.tools.cypher_query import CypherLibrary
        return CypherLibrary
    if name == "build_text2cypher_tool":
        from src.tools.text2cypher import build_text2cypher_tool
        return build_text2cypher_tool
    raise AttributeError(f"module 'src.tools' has no attribute {name!r}")
