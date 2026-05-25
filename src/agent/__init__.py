"""
Agente conversacional Critical Graph RAG.

API pública (compatible con `from src.agent import ...` previo):
  - ChatbotAgent: orquestador principal.
  - get_agent: devuelve singleton (lazy init).
  - init_agent: instancia explícitamente (usar en startup).
  - AgentState: estado del grafo (re-exportado para tests / extensión).
"""

from src.agent.chatbot import ChatbotAgent, get_agent, init_agent
from src.agent.state import AgentState

__all__ = ["AgentState", "ChatbotAgent", "get_agent", "init_agent"]
