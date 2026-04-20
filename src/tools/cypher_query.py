"""
Tool `cypher_query` — ejecución de queries predefinidas.

*Scaffolding.* Se completa en Etapa 3.

Diseño previsto:
  1. Se carga `config/cypher_library.yaml` una sola vez.
  2. El agente elige un `id` de la biblioteca basado en `description` + `when_to_use`.
  3. Se parametriza la query con los `parameters` declarados y se ejecuta vía Neo4j driver.
  4. Devuelve las filas (como list[dict]) al agente.

El scaffolding ya carga la biblioteca y expone un iterador de metadata
para que el planificador la introspeccione.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.config import load_yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CypherQuery:
    id: str
    description: str
    when_to_use: str
    parameters: list[dict[str, Any]]
    returns: str
    cypher: str


class CypherLibrary:
    """Wrapper sobre config/cypher_library.yaml."""

    def __init__(self, path: str = "config/cypher_library.yaml") -> None:
        raw = load_yaml(path)
        self._queries: dict[str, CypherQuery] = {}
        for q in raw.get("queries", []):
            cq = CypherQuery(
                id=q["id"],
                description=q["description"],
                when_to_use=q.get("when_to_use", "").strip(),
                parameters=q.get("parameters", []),
                returns=q.get("returns", ""),
                cypher=q["cypher"].strip(),
            )
            self._queries[cq.id] = cq

    def all(self) -> list[CypherQuery]:
        return list(self._queries.values())

    def get(self, query_id: str) -> CypherQuery:
        return self._queries[query_id]

    def as_catalog(self) -> list[dict[str, Any]]:
        """Resumen legible por LLM (sin el Cypher crudo) para decidir cuál usar."""
        return [
            {
                "id": q.id,
                "description": q.description,
                "when_to_use": q.when_to_use,
                "parameters": q.parameters,
                "returns": q.returns,
            }
            for q in self._queries.values()
        ]


def build_cypher_query_tool(*_args, **_kwargs):
    """TODO (Etapa 3): devolver StructuredTool que ejecute queries del library."""
    raise NotImplementedError("cypher_query tool se implementa en Etapa 3.")
