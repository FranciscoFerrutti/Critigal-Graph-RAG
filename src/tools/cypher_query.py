"""
Tool `cypher_query` — ejecución de queries predefinidas del catálogo.

Flujo:
  1. Carga `config/cypher_library.yaml` (una vez al construir la tool).
  2. El planner LLM ve el catálogo en la `description` de la tool y elige
     `query_id` + completa `parameters`.
  3. La tool valida los parámetros contra la declaración del library,
     aplica defaults y coerciones de tipo (string/int/float).
  4. Ejecuta el Cypher vía un driver de Neo4j obtenido de `get_driver()`.
  5. Devuelve filas como list[dict] con tipos JSON-safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config import load_yaml
from src.neo4j_conn import get_driver
from src.tools.registry import ToolContext, register_tool

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
        if query_id not in self._queries:
            raise KeyError(
                f"query_id '{query_id}' no existe en el library. "
                f"IDs válidos: {sorted(self._queries.keys())}"
            )
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


class CypherQueryArgs(BaseModel):
    query_id: str = Field(
        ...,
        description="ID exacto de la query del catálogo de cypher_library.yaml",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parámetros para la query, en formato dict {nombre: valor}",
    )


_TYPE_COERCERS = {
    "integer": lambda v: int(v) if v is not None else None,
    "int": lambda v: int(v) if v is not None else None,
    "float": lambda v: float(v) if v is not None else None,
    "string": lambda v: str(v) if v is not None else None,
    "boolean": lambda v: bool(v) if v is not None else None,
}


def _coerce_params(cq: CypherQuery, params: dict[str, Any]) -> dict[str, Any]:
    """
    Para cada parámetro declarado:
      - si está presente, intenta coerción al tipo declarado.
      - si falta y es required, lanza ValueError.
      - si falta y es opcional, inyecta el default (puede ser None).
    Parámetros extra (no declarados) se mantienen pass-through.
    """
    out: dict[str, Any] = {k: v for k, v in params.items()}
    for spec in cq.parameters:
        name = spec["name"]
        ptype = spec.get("type", "string")
        required = spec.get("required", True)
        if name in out:
            try:
                coercer = _TYPE_COERCERS.get(ptype)
                if coercer is not None:
                    out[name] = coercer(out[name])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Parámetro '{name}' (tipo declarado: {ptype}) no se pudo "
                    f"convertir desde valor {out[name]!r}: {e}"
                )
        else:
            if required:
                raise ValueError(
                    f"Parámetro requerido '{name}' faltante para query '{cq.id}'."
                )
            out[name] = spec.get("default", None)
    return out


def _jsonify(value: Any) -> Any:
    """Convierte tipos Neo4j (Date, DateTime) a strings serializables."""
    name = type(value).__name__
    if name in {"Date", "DateTime", "Time", "Duration"}:
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _format_catalog_for_prompt(catalog: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for q in catalog:
        params = ", ".join(
            f"{p['name']}:{p.get('type','string')}"
            f"{'' if p.get('required', True) else '?'}"
            for p in q["parameters"]
        )
        lines.append(
            f"• id='{q['id']}' — {q['description']}\n"
            f"  Cuándo: {q['when_to_use'].splitlines()[0] if q['when_to_use'] else '—'}\n"
            f"  Params: ({params}) → Returns: {q['returns']}"
        )
    return "\n".join(lines)


def build_cypher_query_tool(
    library_path: str = "config/cypher_library.yaml",
) -> StructuredTool:
    """
    Construye la StructuredTool `cypher_query` lista para registrar en el agente.
    El catálogo del library queda inyectado en la `description` para que el
    planner LLM pueda elegir un `query_id` válido.
    """
    library = CypherLibrary(library_path)
    catalog = library.as_catalog()
    catalog_text = _format_catalog_for_prompt(catalog)

    def _run(query_id: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            cq = library.get(query_id)
        except KeyError as e:
            return [{"error": str(e)}]
        try:
            params = _coerce_params(cq, parameters or {})
        except ValueError as e:
            return [{"error": str(e)}]
        try:
            with get_driver() as driver, driver.session() as session:
                result = session.run(cq.cypher, **params)
                rows = [_jsonify(dict(record)) for record in result]
            return rows or [{"info": "Sin resultados."}]
        except Exception as e:
            logger.exception("Falló cypher_query id=%s params=%s", query_id, params)
            return [{"error": f"Error ejecutando Cypher: {e}"}]

    description = (
        "Ejecuta una query Cypher predefinida del catálogo. Úsala para preguntas "
        "factuales, agregaciones, conteos, sumas, rankings y comparaciones temporales "
        "donde haya una query con patrón equivalente.\n\n"
        "Para llamarla, pasá `query_id` exacto y un dict `parameters` con los valores. "
        "El query_id debe ser uno de los listados abajo. Si no encontrás match, no "
        "inventes; usá similarity_search o respondé que no podés.\n\n"
        f"Catálogo disponible:\n{catalog_text}"
    )

    return StructuredTool.from_function(
        name="cypher_query",
        description=description,
        args_schema=CypherQueryArgs,
        func=_run,
    )


@register_tool("cypher_query")
def _registry_builder(ctx: ToolContext) -> StructuredTool:
    """Entry-point del registry. Lee `library_path` desde agent_config.paths."""
    library_path = ctx.agent_config.get("paths", {}).get(
        "cypher_library", "config/cypher_library.yaml"
    )
    return build_cypher_query_tool(library_path=library_path)
