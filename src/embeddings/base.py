"""
Interfaz abstracta para proveedores de embeddings y factory de instancias.

Cada proveedor concreto (Google, HuggingFace, OpenAI, ...) implementa
`EmbeddingProvider` devolviendo objetos `Embeddings` de LangChain, para
que el resto del stack (Neo4jVector, retrievers) los consuma sin acoplarse
a un SDK específico.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings

from src.config import get_active_provider_config


class EmbeddingProvider(ABC):
    """
    Adaptador entre la config YAML y los objetos `Embeddings` de LangChain.

    La distinción document/query existe porque modelos como
    `gemini-embedding-001` producen vectores ligeramente distintos según
    el `task_type` (retrieval_document vs retrieval_query), lo que mejora
    el recall en búsquedas cross-lingual.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model: str = config["model"]
        self.dimensions: int = config["dimensions"]
        self.batch_size: int = config.get("batch_size", 64)
        self.api_key_env: str | None = config.get("api_key_env")

    @abstractmethod
    def get_document_embeddings(self) -> Embeddings:
        """Embeddings de documentos (las `notes` del CSV)."""

    @abstractmethod
    def get_query_embeddings(self) -> Embeddings:
        """Embeddings de preguntas del usuario en tiempo de retrieval."""

    @property
    def provider_name(self) -> str:
        return self.config.get("provider", "unknown")


def get_embedding_provider(embeddings_yaml: dict[str, Any]) -> EmbeddingProvider:
    """
    Factory: lee el YAML completo de `config/embeddings.yaml` y devuelve
    la implementación concreta correspondiente al campo `provider:`.
    """
    active = get_active_provider_config(embeddings_yaml)
    provider_name = active["provider"]

    if provider_name == "google":
        from src.embeddings.google import GoogleEmbeddingProvider
        return GoogleEmbeddingProvider(active)
    if provider_name == "huggingface":
        from src.embeddings.huggingface import HuggingFaceEmbeddingProvider
        return HuggingFaceEmbeddingProvider(active)

    raise ValueError(
        f"Proveedor de embeddings desconocido: '{provider_name}'. "
        f"Agregar implementación en src/embeddings/ y registrar en el factory."
    )
