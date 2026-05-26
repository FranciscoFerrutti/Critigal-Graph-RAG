"""
Tool `text2cypher` — fallback LLM para preguntas fuera del library.

*Scaffolding.* Se completa en Etapa 3.

Diseño previsto:
  1. Se construye el prompt con:
       - el graph_schema (labels + propiedades + relaciones)
       - ejemplos few-shot tomados de cypher_library como referencia de estilo
       - la pregunta del usuario
  2. Gemini 2.5 Flash (ver config/agent.yaml) genera la Cypher.
  3. Se valida sintácticamente (EXPLAIN) y se ejecuta.
  4. Se devuelve el resultado + la query generada (para trazabilidad).

Planeo usar `langchain_neo4j.GraphCypherQAChain` como base y agregar
validación + retry con corrección si la primera ejecución falla.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_text2cypher_tool(*_args, **_kwargs):
    """TODO (Etapa 3): devolver StructuredTool que use GraphCypherQAChain."""
    raise NotImplementedError("text2cypher tool se implementa en Etapa 3.")


# Al implementar _run en Etapa 3, agregar antes de ejecutar la Cypher generada:
#   logger.debug("text2cypher: cypher generado=\n%s", generated_cypher)
