"""
Factory de LLMs del agente.

Encapsular la instanciación acá permite cambiar de proveedor (Gemini → Claude,
OpenAI, etc.) tocando solo este módulo. El `agent_config` declara `provider`
y la sub-sección del proveedor activo.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel


def build_planner_llm(agent_config: dict[str, Any], api_key: str | None) -> BaseChatModel:
    """
    Construye el LLM del nodo `planner` según `agent_config.provider`.

    Soporta:
        - google (gemini): usa `langchain_google_genai.ChatGoogleGenerativeAI`.
        - groq: usa `langchain_groq.ChatGroq`.

    Para sumar otro provider: agregar rama acá y la dependencia opcional.
    """
    provider = agent_config.get("provider", "google")

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        planner_cfg = agent_config["google"]["planner"]
        return ChatGoogleGenerativeAI(
            model=planner_cfg["model"],
            temperature=planner_cfg["temperature"],
            google_api_key=api_key,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        planner_cfg = agent_config["groq"]["planner"]
        return ChatGroq(
            model=planner_cfg["model"],
            temperature=planner_cfg["temperature"],
            groq_api_key=api_key,
        )

    raise ValueError(
        f"Provider LLM desconocido: '{provider}'. "
        f"Agregar rama en src/agent/llm.build_planner_llm."
    )
