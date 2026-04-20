"""
Tools expuestas al agente.

Tres herramientas:
  - similarity_search : búsqueda vectorial sobre Event.notes (funcional).
  - cypher_query      : queries predefinidas de config/cypher_library.yaml (scaffolding).
  - text2cypher       : fallback — LLM traduce pregunta -> Cypher (scaffolding).

Los submódulos se importan perezosamente para no forzar la dependencia de
LangChain cuando sólo se necesita, por ejemplo, `CypherLibrary` (puro YAML).
"""

__all__ = ["build_similarity_search_tool", "CypherLibrary"]


def __getattr__(name: str):
    if name == "build_similarity_search_tool":
        from src.tools.similarity_search import build_similarity_search_tool
        return build_similarity_search_tool
    if name == "CypherLibrary":
        from src.tools.cypher_query import CypherLibrary
        return CypherLibrary
    raise AttributeError(f"module 'src.tools' has no attribute {name!r}")
