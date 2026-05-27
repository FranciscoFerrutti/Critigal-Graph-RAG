"""
Tool `text2cypher` — fallback LLM para preguntas fuera del library.

Flujo:
  1. Construye un prompt con el schema del grafo + ejemplos few-shot del
     `cypher_library.yaml` + la pregunta del usuario.
  2. Pide al LLM (Gemini 2.5 Flash por defecto, ver `config/agent.yaml`)
     que devuelva solo la Cypher.
  3. Sanitiza la salida (quita code fences, normaliza), bloquea operaciones
     de escritura, valida con `EXPLAIN` y ejecuta.
  4. Si EXPLAIN o la ejecución fallan, reintenta una vez pidiéndole al LLM
     que corrija usando el mensaje de error como contexto.
  5. Devuelve filas como list[dict]. En caso de error o sin resultados,
     adjunta `generated_cypher` para trazabilidad.

No usamos `GraphCypherQAChain` para mantener control sobre el prompt,
el guard de read-only y la política de reintentos.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config import load_yaml, resolve_api_key
from src.neo4j_conn import get_driver
from src.schema import GraphSchema
from src.tools.cypher_query import CypherLibrary, _jsonify
from src.tools.registry import ToolContext, register_tool

logger = logging.getLogger(__name__)


# Tokens Cypher que indican operaciones de escritura. La query generada NO
# debe contenerlos: el agente es de solo-lectura sobre el KG.
_WRITE_TOKENS = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD CSV",
)


class Text2CypherArgs(BaseModel):
    """Argumentos que el agente le pasa a la tool."""

    question: str = Field(
        ...,
        description=(
            "Pregunta del usuario en lenguaje natural (ES/EN) que se traducirá "
            "a Cypher contra el schema del KG."
        ),
    )


# ---------------------------------------------------------------------------
# Renderizado de contexto para el prompt
# ---------------------------------------------------------------------------

def _render_attr(a: Any) -> str:
    """Render `name:type` o `name:type('FORMAT')` si el atributo declara `format`."""
    if getattr(a, "format", None):
        return f"{a.name}:{a.type}('{a.format}')"
    return f"{a.name}:{a.type}"


def _render_schema(schema: GraphSchema) -> str:
    """
    Convierte `GraphSchema` en texto compacto para el prompt:
    lista de nodos con sus propiedades y relaciones direccionadas.
    Si un atributo declara `format` (ej. fechas string), se anota inline
    para que el LLM no asuma tipos nativos.
    """
    lines: list[str] = ["NODOS:"]
    for node in schema.node_types.values():
        attrs = ", ".join(_render_attr(a) for a in node.attributes)
        lines.append(f"  ({node.label} {{{attrs}}})")
    lines.append("")
    lines.append("RELACIONES (dirección from -[type]-> to):")
    for rel in schema.relationship_types.values():
        from_label = schema.node(rel.from_node).label
        to_label = schema.node(rel.to_node).label
        attrs = ", ".join(_render_attr(a) for a in rel.attributes)
        props = f" {{{attrs}}}" if attrs else ""
        lines.append(f"  ({from_label})-[:{rel.type}{props}]->({to_label})")
    return "\n".join(lines)


def _render_few_shot_examples(library: CypherLibrary, n: int = 4) -> str:
    """Toma N queries del library como ejemplos few-shot para guiar el estilo."""
    examples = library.all()[:n]
    if not examples:
        return "(sin ejemplos disponibles)"
    blocks: list[str] = []
    for q in examples:
        blocks.append(f"Pregunta tipo: {q.description}\nCypher:\n{q.cypher}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# LLM factory (local — paralelo a src/agent/llm.py para no introducir
# dependencia circular tools -> agent)
# ---------------------------------------------------------------------------

def _build_text2cypher_llm(
    agent_config: dict[str, Any], api_key: str | None
) -> BaseChatModel:
    provider = agent_config.get("provider", "google")
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        cfg = agent_config["google"]["text2cypher"]
        return ChatGoogleGenerativeAI(
            model=cfg["model"],
            temperature=cfg["temperature"],
            google_api_key=api_key,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        cfg = agent_config["groq"]["text2cypher"]
        return ChatGroq(
            model=cfg["model"],
            temperature=cfg["temperature"],
            groq_api_key=api_key,
        )
    raise ValueError(
        f"Provider LLM desconocido: '{provider}'. "
        f"Agregar rama en src/tools/text2cypher._build_text2cypher_llm."
    )


# ---------------------------------------------------------------------------
# Sanitización + validación de la Cypher generada
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:cypher)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_PREFIX_RE = re.compile(r"^(cypher|query)\s*:\s*", re.IGNORECASE)


def _extract_cypher(text: str) -> str:
    """Limpia la salida del LLM: quita code fences, prefijos y `;` final."""
    s = text.strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    s = _PREFIX_RE.sub("", s).strip()
    s = s.rstrip(";").strip()
    return s


def _is_read_only(cypher: str) -> bool:
    """Heurística: rechaza queries con tokens de escritura fuera de strings."""
    # Ocultar literales string para no levantar falsos positivos.
    no_literals = re.sub(r"('[^']*'|\"[^\"]*\")", "''", cypher)
    upper = no_literals.upper()
    for token in _WRITE_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", upper):
            return False
    return True


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_GEN_PROMPT_TEMPLATE = """\
Sos un experto en Cypher (Neo4j) y traducís preguntas en lenguaje natural a
queries Cypher sobre el siguiente Knowledge Graph de eventos ACLED.

SCHEMA DEL GRAFO:
{schema}

EJEMPLOS REALES DE QUERIES SOBRE ESTE GRAFO:
{examples}

REGLAS DURAS:
- Devolvé SOLO la query Cypher. Sin explicaciones, sin markdown, sin ```.
- La query debe ser de SOLO LECTURA: nunca uses CREATE, MERGE, DELETE, SET,
  REMOVE, DROP ni LOAD CSV.
- Usá los labels, tipos de relación y nombres de propiedades EXACTOS del
  schema (case-sensitive). No inventes propiedades ni labels.
- Si la pregunta menciona Gaza o Cisjordania usá Location.admin1
  (valores típicos: 'Gaza Strip', 'West Bank'). Para países completos usá
  Location.country (ej: 'Israel', 'Palestine').
- Event.event_date es tipo string en formato 'YYYY-MM-DD'. Comparar contra
  string literal: e.event_date = '2023-10-07'. NUNCA envolver con date().
  Para rangos usar comparación lexicográfica: e.event_date >= '2023-10-01'.
- Para agrupar por mes preferí (Event)-[:IN_MONTH]->(Month) y Month.month_id
  ('YYYY-MM').
- Devolvé columnas con alias claros (AS event_count, AS actor_name, etc.).
- Cuando devuelvas listas, incluí LIMIT (default LIMIT 25).
- Para contar nodos únicos usá count(DISTINCT ...).

PREGUNTA DEL USUARIO:
{question}
"""


_REPAIR_PROMPT_TEMPLATE = """\
La query Cypher anterior falló. Generá una nueva que corrija el problema.

SCHEMA DEL GRAFO:
{schema}

PREGUNTA ORIGINAL:
{question}

CYPHER GENERADA (ROTA):
{previous_cypher}

ERROR DE NEO4J:
{error}

Devolvé SOLO la nueva query Cypher, sin explicaciones ni markdown. Mantené
las reglas: solo lectura, labels/propiedades exactos del schema, alias
claros, LIMIT cuando aplique.
"""


# ---------------------------------------------------------------------------
# Builder principal
# ---------------------------------------------------------------------------

def build_text2cypher_tool(
    agent_config: dict[str, Any] | None = None,
    graph_schema: GraphSchema | None = None,
    cypher_library: CypherLibrary | None = None,
    max_retries: int = 1,
) -> StructuredTool:
    """
    Construye la tool `text2cypher` lista para registrar en el agente.

    Args:
        agent_config: dict de `config/agent.yaml`. Lee si es None.
        graph_schema: schema parseado. Carga desde YAML si es None.
        cypher_library: library con few-shots. Carga desde `paths.cypher_library`
                        si es None.
        max_retries: cantidad de reintentos con corrección si la primera
                     ejecución (o el EXPLAIN) falla.
    """
    agent_cfg = agent_config or load_yaml("config/agent.yaml")
    schema = graph_schema or GraphSchema.from_yaml("config/graph_schema.yaml")
    library = cypher_library or CypherLibrary(
        agent_cfg.get("paths", {}).get("cypher_library", "config/cypher_library.yaml")
    )

    provider = agent_cfg.get("provider", "google")
    api_key = resolve_api_key(agent_cfg.get(provider, {}).get("api_key_env"))
    llm = _build_text2cypher_llm(agent_cfg, api_key)

    schema_text = _render_schema(schema)
    examples_text = _render_few_shot_examples(library, n=4)

    def _llm_to_cypher(prompt: str) -> str:
        msg = llm.invoke([HumanMessage(content=prompt)])
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return _extract_cypher(content)

    def _generate(question: str) -> str:
        return _llm_to_cypher(
            _GEN_PROMPT_TEMPLATE.format(
                schema=schema_text,
                examples=examples_text,
                question=question,
            )
        )

    def _repair(question: str, previous: str, error: str) -> str:
        return _llm_to_cypher(
            _REPAIR_PROMPT_TEMPLATE.format(
                schema=schema_text,
                question=question,
                previous_cypher=previous,
                error=error,
            )
        )

    def _run(question: str) -> list[dict[str, Any]]:
        logger.debug("text2cypher: pregunta=%r", question)

        try:
            cypher = _generate(question)
        except Exception as e:
            logger.exception("text2cypher: error generando cypher")
            return [{"error": f"No se pudo generar Cypher: {e}"}]

        attempt = 0
        while True:
            logger.debug(
                "text2cypher: intento=%d cypher=\n%s", attempt, cypher
            )

            if not _is_read_only(cypher):
                msg = (
                    "La query generada incluye operaciones de escritura "
                    "(CREATE/MERGE/DELETE/SET/REMOVE/DROP/LOAD CSV). Rechazada."
                )
                logger.warning("text2cypher: %s\n%s", msg, cypher)
                if attempt >= max_retries:
                    return [{"error": msg, "generated_cypher": cypher}]
                cypher = _repair(question, cypher, msg)
                attempt += 1
                continue

            try:
                with get_driver() as driver, driver.session() as session:
                    # 1) Validar sintaxis con EXPLAIN (no ejecuta el plan).
                    session.run(f"EXPLAIN {cypher}").consume()
                    # 2) Ejecutar.
                    result = session.run(cypher)
                    rows = [_jsonify(dict(record)) for record in result]
            except Exception as e:
                err = str(e)
                logger.warning(
                    "text2cypher: fallo (intento %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    err,
                )
                if attempt >= max_retries:
                    return [{
                        "error": f"Error ejecutando Cypher: {err}",
                        "generated_cypher": cypher,
                    }]
                cypher = _repair(question, cypher, err)
                attempt += 1
                continue

            if not rows:
                logger.warning(
                    "text2cypher: 0 filas para cypher=\n%s", cypher
                )
                return [{
                    "info": "Sin resultados.",
                    "generated_cypher": cypher,
                }]
            return rows

    description = (
        "Fallback. Traduce una pregunta en lenguaje natural a Cypher contra el "
        "KG ACLED y la ejecuta. Úsala SOLO cuando ninguna query del catálogo "
        "de `cypher_query` matchea el patrón de la pregunta. Devuelve filas "
        "como list[dict]; en caso de error o sin resultados incluye "
        "`generated_cypher` para trazabilidad."
    )

    return StructuredTool.from_function(
        name="text2cypher",
        description=description,
        args_schema=Text2CypherArgs,
        func=_run,
    )


@register_tool("text2cypher")
def _registry_builder(ctx: ToolContext) -> StructuredTool:
    """Entry-point del registry. Lee `library_path` desde agent_config.paths."""
    library_path = ctx.agent_config.get("paths", {}).get(
        "cypher_library", "config/cypher_library.yaml"
    )
    return build_text2cypher_tool(
        agent_config=ctx.agent_config,
        graph_schema=ctx.graph_schema,
        cypher_library=CypherLibrary(library_path),
    )
