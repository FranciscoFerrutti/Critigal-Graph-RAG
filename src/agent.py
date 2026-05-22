"""
Agente conversacional LangGraph para Critical Graph RAG.

Orquesta el flujo:
  1. Recibe pregunta del usuario (ES/EN).
  2. Elige si usar similarity_search, cypher_query, o text2cypher.
  3. Ejecuta la tool seleccionada.
  4. Formula la respuesta basándose SOLO en los resultados de la tool.
  5. Maneja casos donde ninguna tool puede resolver la pregunta.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from src.config import load_yaml, resolve_api_key
from src.schema import GraphSchema
from src.tools.cypher_query import build_cypher_query_tool
from src.tools.similarity_search import build_similarity_search_tool

logger = logging.getLogger(__name__)


# =============================================================================
# State Management
# =============================================================================

class AgentState(BaseModel):
    """Estado compartido del agente conversacional."""

    messages: list[BaseMessage]
    tool_results: dict[str, Any] | None = None
    response: str | None = None
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Tool Management
# =============================================================================

def _build_tools(
    agent_config: dict[str, Any],
    embeddings_config: dict[str, Any],
    graph_schema: GraphSchema,
) -> list[StructuredTool]:
    """
    Construye la lista de tools disponibles para el agente.

    Tools activas:
      - cypher_query: ejecuta queries predefinidas del cypher_library.yaml.
      - similarity_search: búsqueda vectorial sobre Event.notes.

    `text2cypher` queda como scaffolding (mejora futura).
    """
    tools: list[StructuredTool] = []

    # Cypher query: catálogo de queries predefinidas
    try:
        library_path = agent_config.get("paths", {}).get(
            "cypher_library", "config/cypher_library.yaml"
        )
        cypher_query = build_cypher_query_tool(library_path=library_path)
        tools.append(cypher_query)
        logger.info("✓ Tool 'cypher_query' cargada")
    except Exception as e:
        logger.warning(f"⚠ No se pudo cargar cypher_query: {e}")

    # Similarity search: búsqueda vectorial sobre Event.notes
    try:
        similarity_search = build_similarity_search_tool(
            agent_config=agent_config,
            embeddings_config=embeddings_config,
            graph_schema=graph_schema,
        )
        tools.append(similarity_search)
        logger.info("✓ Tool 'similarity_search' cargada")
    except Exception as e:
        logger.warning(f"⚠ No se pudo cargar similarity_search: {e}")

    return tools


# =============================================================================
# Prompt System
# =============================================================================

SYSTEM_PROMPT_ES = """
Eres un asistente experto en análisis de conflictos y datos de eventos ACLED.
El dataset cargado cubre eventos en Israel durante 2023 (4259 eventos, 6 distritos:
HaDarom, HaMerkaz, HaZafon, Haifa, Jerusalem, Tel Aviv).

REGLAS CRÍTICAS:
1. Responde SOLO basándote en los resultados que las tools devuelven. No inventes.
2. Para preguntas factuales (conteos, sumas, rankings, comparaciones temporales),
   USA `cypher_query`: elegí un `query_id` del catálogo y completá `parameters`.
3. Para preguntas descriptivas o de búsqueda por contenido ("¿qué pasó cuando…?",
   "describí los eventos en…"), USA `similarity_search`.
4. Si una tool devuelve `[{"error": "..."}]`, leé el error y reintentá con
   `query_id` válido o parámetros corregidos. Si tras dos intentos no resolvés,
   cambiá de tool o respondé honestamente que no podés.
5. Si la tool devuelve `[{"info": "Sin resultados."}]`, respondé:
   "No encontré datos para esa consulta en el dataset."
6. Si la pregunta es fuera de dominio, respondé:
   "Esa pregunta está fuera de mi área de expertise. Solo respondo sobre eventos
   ACLED (Israel 2023)."
7. Respondé en el mismo idioma que la pregunta.
8. Resumí los resultados de forma clara y estructurada (números con unidad,
   listas con bullets cuando aplique).

TOOLS DISPONIBLES:
- cypher_query: Query Cypher predefinida del catálogo. Para conteos, sumas,
  rankings y comparaciones temporales con patrón conocido. Args: query_id, parameters.
- similarity_search: Búsqueda vectorial sobre Event.notes. Para preguntas
  descriptivas / búsqueda semántica por contenido. Args: query, top_k.
"""

SYSTEM_PROMPT_EN = """
You are an expert assistant in conflict analysis and ACLED event data.
The loaded dataset covers events in Israel during 2023 (4259 events, 6 districts:
HaDarom, HaMerkaz, HaZafon, Haifa, Jerusalem, Tel Aviv).

CRITICAL RULES:
1. Answer ONLY based on what tools return. Do not invent.
2. For factual questions (counts, sums, rankings, time comparisons),
   USE `cypher_query`: pick a `query_id` from the catalog and fill `parameters`.
3. For descriptive or content search questions ("what happened when…?",
   "describe the events in…"), USE `similarity_search`.
4. If a tool returns `[{"error": "..."}]`, read the error and retry with a
   valid query_id or corrected params. If unresolved after two tries,
   switch tools or honestly say you cannot answer.
5. If the tool returns `[{"info": "Sin resultados."}]`, answer:
   "I couldn't find data for that query in the dataset."
6. If the question is out of domain, answer:
   "That question is outside my expertise. I only answer ACLED events (Israel 2023)."
7. Answer in the same language as the question.
8. Summarize clearly (numbers with units, bullets for lists when applicable).

AVAILABLE TOOLS:
- cypher_query: Predefined Cypher from the catalog. For counts, sums, rankings,
  time comparisons with known patterns. Args: query_id, parameters.
- similarity_search: Vector search over Event.notes. For descriptive /
  semantic content questions. Args: query, top_k.
"""


def _detect_language(text: str) -> str:
    """Detecta si un texto está en español o inglés."""
    # Simple heuristic: contar palabras comunes en cada idioma
    es_words = {"qué", "por", "para", "como", "donde", "cuando", "el", "la", "los", "las"}
    en_words = {"what", "why", "how", "where", "when", "the", "a", "is", "are", "have"}

    text_lower = text.lower()
    es_score = sum(1 for w in es_words if w in text_lower)
    en_score = sum(1 for w in en_words if w in text_lower)

    return "es" if es_score > en_score else "en"


def _get_system_prompt(language: str) -> str:
    """Obtiene el prompt del sistema en el idioma detectado."""
    return SYSTEM_PROMPT_ES if language == "es" else SYSTEM_PROMPT_EN


# =============================================================================
# Agent Builder
# =============================================================================

class ChatbotAgent:
    """Orquestador principal del agente conversacional."""

    def __init__(self, config_dir: str = "config") -> None:
        """
        Inicializa el agente cargando configuración y construyendo tools.

        Args:
            config_dir: Ruta al directorio de configuración (default: "config")
        """
        self.config_dir = config_dir

        # Cargar configuraciones de YAML
        self.agent_config = load_yaml(f"{config_dir}/agent.yaml")
        self.embeddings_config = load_yaml(f"{config_dir}/embeddings.yaml")
        self.graph_schema = GraphSchema.from_yaml(f"{config_dir}/graph_schema.yaml")

        # Resolver API key de Google
        api_key_env = self.agent_config["google"]["api_key_env"]
        self.google_api_key = resolve_api_key(api_key_env)

        # Instanciar LLM del planner
        planner_config = self.agent_config["google"]["planner"]
        self.planner_llm = self._build_llm(
            model=planner_config["model"],
            temperature=planner_config["temperature"],
        )

        # Construir tools
        self.tools = _build_tools(
            self.agent_config,
            self.embeddings_config,
            self.graph_schema,
        )

        # Construir grafo del agente
        self.graph = self._build_graph()
        self.runnable = self.graph.compile()

        logger.info(f"✓ Agente inicializado con {len(self.tools)} tool(s)")

    def _build_llm(self, model: str, temperature: float) -> BaseChatModel:
        """Construye la instancia de Google Generative AI LLM."""
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=self.google_api_key,
        )

    def _build_graph(self) -> StateGraph:
        """
        Construye el grafo de estado del agente (LangGraph).
        
        Patrón ReAct (Reasoning + Acting):
        1. process: LLM piensa y decide si usar tools.
        2. should_continue: Condicional - ¿hay tool calls?
        3. tools: Ejecuta las tools.
        4. loop back a process: LLM recibe resultados y genera respuesta final.
        5. finalize: Extrae la respuesta final cuando el LLM decide parar.
        """
        from langgraph.graph import START, END

        graph = StateGraph(AgentState)

        # Nodos
        graph.add_node("process", self._process_message)
        graph.add_node("finalize", self._finalize_response)

        if self.tools:
            tool_node = ToolNode(self.tools)
            graph.add_node("tools", tool_node)

        # Puntos de entrada y salida
        graph.add_edge(START, "process")

        if self.tools:
            # Condicional: ¿hay tool calls en la respuesta del LLM?
            graph.add_conditional_edges(
                "process",
                self._should_use_tools,
                {
                    "tools": "tools",      # Ejecutar tool
                    "finalize": "finalize",  # Ir a respuesta final
                },
            )
            # Después de ejecutar tools, volver a process para que LLM refine la respuesta
            graph.add_edge("tools", "process")
        else:
            # Sin tools, ir directo a finalizar
            graph.add_edge("process", "finalize")

        # Finalizar y salir
        graph.add_edge("finalize", END)

        return graph

    def _should_use_tools(self, state: AgentState) -> str:
        """
        Condicional que determina si el último mensaje del LLM contiene
        tool calls. Si es así, ir a "tools"; si no, ir a "finalize".
        """
        last_msg = state.messages[-1]

        # Verificar si el mensaje tiene tool_calls
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        
        return "finalize"

    def _process_message(self, state: AgentState) -> dict[str, Any]:
        """
        Nodo del grafo: procesa el mensaje y orquesta la respuesta.

        Dos modos:
          - Primera iteración (sin ToolMessages aún): bind tools y dejar que el
            planner decida qué tool usar.
          - Segunda iteración (ya hay ToolMessages): NO reenviar el historial
            (gemini-2.5-flash devuelve AIMessage con content="" + tool_calls,
            cuyo segundo envío rompe con `contents are required`). En su lugar,
            armamos un prompt nuevo con la pregunta original + el resultado
            crudo de la tool y pedimos síntesis sin tools.
        """
        # Pregunta original (primer HumanMessage).
        user_text = ""
        for m in state.messages:
            if m.__class__.__name__ == "HumanMessage":
                user_text = m.content if isinstance(m.content, str) else str(m.content)
                break

        language = _detect_language(user_text)
        system_prompt = _get_system_prompt(language)

        # Resultados de tools acumulados en este turno.
        tool_msgs = [m for m in state.messages if m.__class__.__name__ == "ToolMessage"]
        tool_calls_seen = []
        for m in state.messages:
            calls = getattr(m, "tool_calls", None) or []
            for tc in calls:
                tool_calls_seen.append(
                    tc if isinstance(tc, dict) else {"name": getattr(tc, "name", ""), "args": getattr(tc, "args", {})}
                )

        if not tool_msgs:
            # === Primera iteración: planner con tools ===
            llm_with_tools = self.planner_llm.bind_tools(self.tools) if self.tools else self.planner_llm
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]
            response = llm_with_tools.invoke(messages)
        else:
            # === Síntesis: pregunta + resultado(s) de tool(s) → respuesta ===
            tool_context_lines = []
            for tc, tm in zip(tool_calls_seen, tool_msgs):
                tool_context_lines.append(
                    f"Tool `{tc.get('name','?')}` llamada con args={tc.get('args',{})}\n"
                    f"Resultado: {tm.content}"
                )
            tool_context = "\n\n".join(tool_context_lines) or "(sin resultados de tool)"
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

        new_messages = state.messages + [response]
        return {"messages": new_messages}

    def _finalize_response(self, state: AgentState) -> dict[str, Any]:
        """
        Nodo del grafo: extrae la respuesta final del agente.

        Busca el último AIMessage en el historial que contenga texto
        (que será la respuesta del LLM después de procesar tools).
        """
        # Buscar hacia atrás en los messages para encontrar un AIMessage con contenido de texto
        for msg in reversed(state.messages):
            # Buscar AIMessage (de tipo langchain_core.messages.ai.AIMessage)
            if msg.__class__.__name__ == "AIMessage":
                if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
                    return {"response": msg.content}

        # Fallback: si no encontramos nada, intentar con el último mensaje
        last_msg = state.messages[-1]
        if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
            response_text = last_msg.content
        else:
            # Último fallback
            response_text = str(last_msg.content) if hasattr(last_msg, "content") else ""

        return {"response": response_text or "No se pudo generar una respuesta."}

    def invoke(self, user_message: str) -> str:
        """
        API principal: procesa un mensaje de usuario y devuelve la respuesta.

        Args:
            user_message: Pregunta del usuario (ES/EN).

        Returns:
            Respuesta del agente (en el idioma de la pregunta).
        """
        try:
            return self.invoke_trace(user_message)["response"]
        except Exception as e:
            logger.error(f"Error en agente: {e}", exc_info=True)
            language = _detect_language(user_message)
            if language == "es":
                return f"Lo siento, hubo un error procesando tu pregunta: {str(e)}"
            else:
                return f"I'm sorry, there was an error processing your question: {str(e)}"

    def invoke_trace(self, user_message: str) -> dict[str, Any]:
        """
        Variante de `invoke` que devuelve también la traza de tools llamadas.

        Returns:
            dict con:
              - response: str
              - tool_calls: list[{"name": str, "args": dict}]
              - messages_count: int
        """
        initial_state = AgentState(messages=[HumanMessage(content=user_message)])
        final_state = self.runnable.invoke(initial_state)

        response = final_state.get("response") or final_state.get("error", "") or "No se pudo procesar tu pregunta."

        tool_calls: list[dict[str, Any]] = []
        for msg in final_state.get("messages", []):
            calls = getattr(msg, "tool_calls", None) or []
            for tc in calls:
                # langchain ToolCall puede ser dict o objeto
                if isinstance(tc, dict):
                    tool_calls.append({"name": tc.get("name", ""), "args": tc.get("args", {})})
                else:
                    tool_calls.append({"name": getattr(tc, "name", ""), "args": getattr(tc, "args", {})})

        return {
            "response": response,
            "tool_calls": tool_calls,
            "messages_count": len(final_state.get("messages", [])),
        }


# =============================================================================
# Singleton global (para no reinicializar en cada request)
# =============================================================================

_agent_instance: ChatbotAgent | None = None


def get_agent() -> ChatbotAgent:
    """Obtiene la instancia singleton del agente."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ChatbotAgent()
    return _agent_instance


def init_agent(config_dir: str = "config") -> ChatbotAgent:
    """Inicializa el agente (llamar en startup de FastAPI)."""
    global _agent_instance
    _agent_instance = ChatbotAgent(config_dir=config_dir)
    return _agent_instance
