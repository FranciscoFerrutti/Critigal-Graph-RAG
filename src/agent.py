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

from langchain_core.language_model import BaseLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from src.config import load_yaml, resolve_api_key
from src.schema import GraphSchema
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

    Por ahora: similarity_search está totalmente implementada.
    Las demás (cypher_query, text2cypher) están en scaffolding.
    """
    tools: list[StructuredTool] = []

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

    # TODO (Etapa 3):
    # - cypher_query: ejecutar queries predefinidas
    # - text2cypher: generar Cypher dinámico con LLM

    return tools


# =============================================================================
# Prompt System
# =============================================================================

SYSTEM_PROMPT_ES = """
Eres un asistente experto en análisis de conflictos y datos de eventos ACLED.
Tu objetivo es responder preguntas sobre eventos de protesta, violencia y activismo.

REGLAS CRÍTICAS:
1. Responde SOLO basándote en los resultados que las tools devuelven.
2. No inventess información que no esté en los resultados.
3. Si una herramienta devuelve resultados, úsalos. Si no hay resultados, responde:
   "Lo siento, no encontré eventos que se ajusten a tu pregunta. Intenta reformularla."
4. Si ninguna tool es aplicable (la pregunta es fuera de dominio), responde:
   "Lo siento, esa pregunta está fuera de mi área de expertise. Solo puedo responder
   sobre eventos de protesta, violencia y activismo documentados en ACLED."
5. Siempre responde en el mismo idioma que se te hizo la pregunta.
6. Resume los resultados de forma clara y estructurada.

TOOLS DISPONIBLES:
- similarity_search: Para buscar eventos por descripción de contenido.
  Úsala cuando la pregunta pida describir eventos, buscar por contexto,
  o cuando no haya una query Cypher precisa disponible.
"""

SYSTEM_PROMPT_EN = """
You are an expert assistant in conflict analysis and ACLED event data.
Your goal is to answer questions about protest events, violence, and activism.

CRITICAL RULES:
1. Answer ONLY based on the results that tools return.
2. Do not invent information that is not in the results.
3. If a tool returns results, use them. If there are no results, respond:
   "I'm sorry, I couldn't find events matching your question. Please try rephrasing it."
4. If no tool is applicable (question out of domain), respond:
   "I'm sorry, that question is outside my area of expertise. I can only answer
   about protest, violence, and activism events documented in ACLED."
5. Always respond in the same language as the question was asked.
6. Summarize results clearly and in a structured format.

AVAILABLE TOOLS:
- similarity_search: To search events by content description.
  Use it when the question asks to describe events, search by context,
  or when there is no precise Cypher query available.
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

    def _build_llm(self, model: str, temperature: float) -> BaseLLM:
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
        Nodo del grafo: procesa el mensaje y decide qué tool usar.

        El LLM recibe el mensaje del usuario y, mediante prompt engineering,
        describe qué tool sería la mejor opción.
        """
        user_msg = state.messages[-1]
        user_text = user_msg.content if isinstance(user_msg.content, str) else str(user_msg.content)

        # Detectar idioma
        language = _detect_language(user_text)
        system_prompt = _get_system_prompt(language)

        # Bind tools al LLM
        if self.tools:
            llm_with_tools = self.planner_llm.bind_tools(self.tools)
        else:
            llm_with_tools = self.planner_llm

        # Construir messages: system + historial
        messages = [SystemMessage(content=system_prompt)] + state.messages

        # Invocar LLM
        response = llm_with_tools.invoke(messages)

        # Actualizar state con la respuesta del LLM
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
            # Construir state inicial
            initial_state = AgentState(messages=[HumanMessage(content=user_message)])

            # Invocar grafo
            final_state = self.runnable.invoke(initial_state)

            # Extraer respuesta
            response = final_state.get("response") or final_state.get("error", "")
            if not response:
                # Fallback
                response = "No se pudo procesar tu pregunta."

            return response

        except Exception as e:
            logger.error(f"Error en agente: {e}", exc_info=True)
            # Determinar idioma para respuesta de error
            language = _detect_language(user_message)
            if language == "es":
                return f"Lo siento, hubo un error procesando tu pregunta: {str(e)}"
            else:
                return f"I'm sorry, there was an error processing your question: {str(e)}"


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
