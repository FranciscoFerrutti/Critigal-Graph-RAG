"""
Tool `similarity_search` — búsqueda vectorial sobre Event.notes.

Flujo:
  1. Embebe la pregunta con `task_type=retrieval_query`.
  2. Busca los top-k Events más cercanos en el índice vectorial de Neo4j.
  3. (Opcional) expande K saltos de vecindario devolviendo el subgrafo.

Usa `langchain_neo4j.Neo4jVector.from_existing_index` para reutilizar el
índice creado por `05_load_neo4j.py`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config import load_yaml
from src.embeddings import get_embedding_provider
from src.schema import GraphSchema
from src.tools.registry import ToolContext, register_tool

logger = logging.getLogger(__name__)


class SimilaritySearchArgs(BaseModel):
    """Argumentos que el agente le pasa a la tool."""
    query: str = Field(..., description="Pregunta del usuario en lenguaje natural (ES o EN).")
    top_k: int = Field(10, description="Cantidad de Events a devolver. Default: 10.")


@dataclass
class SimilaritySearchResult:
    event_id: str
    score: float
    notes: str
    event_date: str | None
    country: str | None
    admin1: str | None


def build_similarity_search_tool(
    agent_config: dict[str, Any] | None = None,
    embeddings_config: dict[str, Any] | None = None,
    graph_schema: GraphSchema | None = None,
) -> StructuredTool:
    """
    Construye una LangChain tool lista para registrar en el agente.
    Lee los YAML del proyecto si no se pasan explícitamente (útil para tests).
    """
    agent_cfg = agent_config or load_yaml("config/agent.yaml")
    emb_cfg = embeddings_config or load_yaml("config/embeddings.yaml")
    schema = graph_schema or GraphSchema.from_yaml("config/graph_schema.yaml")

    provider = get_embedding_provider(emb_cfg)
    query_embeddings = provider.get_query_embeddings()
    default_top_k = int(agent_cfg.get("retrieval", {}).get("top_k", 10))
    vi = schema.vector_index

    def _run(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        from langchain_neo4j import Neo4jVector

        vector_store = Neo4jVector.from_existing_index(
            embedding=query_embeddings,
            url=os.environ["NEO4J_URI"],
            username=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ["NEO4J_PASSWORD"],
            index_name=vi.name,
            node_label=vi.node_label,
            embedding_node_property=vi.property,
            text_node_property="notes",
        )
        k = top_k or default_top_k
        hits = vector_store.similarity_search_with_score(query, k=k)
        results: list[dict[str, Any]] = []
        for doc, score in hits:
            results.append({
                "event_id": doc.metadata.get("event_id"),
                "score": float(score),
                "notes": doc.page_content,
                "event_date": doc.metadata.get("event_date"),
                "country": doc.metadata.get("country"),
                "admin1": doc.metadata.get("admin1"),
            })
        return results

    return StructuredTool.from_function(
        name="similarity_search",
        description=(
            "Busca eventos ACLED semánticamente parecidos a una pregunta en lenguaje natural. "
            "Devuelve hasta top_k eventos con sus notes, fecha y ubicación. "
            "Úsala cuando la pregunta pida describir qué sucedió, buscar eventos por contenido "
            "o cuando no haya una query estructurada precisa en cypher_library."
        ),
        args_schema=SimilaritySearchArgs,
        func=_run,
    )


@register_tool("similarity_search")
def _registry_builder(ctx: ToolContext) -> StructuredTool:
    """Entry-point del registry. Delega al builder principal con ctx desempaquetado."""
    return build_similarity_search_tool(
        agent_config=ctx.agent_config,
        embeddings_config=ctx.embeddings_config,
        graph_schema=ctx.graph_schema,
    )
