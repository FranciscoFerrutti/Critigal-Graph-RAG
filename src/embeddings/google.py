"""
Implementación del proveedor de embeddings de Google (Gemini).

Usa `langchain-google-genai` para exponer `Embeddings` cross-lingual
con `task_type` distinto en documento y query.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.embeddings import Embeddings

from src.config import resolve_api_key
from src.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class GoogleEmbeddingProvider(EmbeddingProvider):
    """
    Envuelve `GoogleGenerativeAIEmbeddings` con dos instancias: una
    configurada con task_type='retrieval_document' (para las `notes`)
    y otra con 'retrieval_query' (para la pregunta del usuario).

    `gemini-embedding-001` soporta Matryoshka: 768 / 1536 / 3072.
    La dimensión se pasa vía `output_dimensionality`.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        # Exportar la API key al env con el nombre que espera el SDK de Google.
        # `resolve_api_key` valida que esté definida.
        key = resolve_api_key(self.api_key_env)
        if key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = key
        self._doc: Embeddings | None = None
        self._query: Embeddings | None = None

    def _make(self, task_type: str) -> Embeddings:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # El SDK acepta el nombre "models/<nombre>" o el nombre pelado.
        model_id = self.model if self.model.startswith("models/") else f"models/{self.model}"
        logger.debug(
            "Creando GoogleGenerativeAIEmbeddings: model=%s dims=%d task_type=%s",
            model_id, self.dimensions, task_type,
        )
        return GoogleGenerativeAIEmbeddings(
            model=model_id,
            task_type=task_type,
            # output_dimensionality controla Matryoshka. Si la versión del SDK
            # no lo expone, Gemini devuelve la dimensión default del modelo;
            # ajustar requirements.txt si es necesario.
            output_dimensionality=self.dimensions,
        )

    def get_document_embeddings(self) -> Embeddings:
        if self._doc is None:
            self._doc = self._make("retrieval_document")
        return self._doc

    def get_query_embeddings(self) -> Embeddings:
        if self._query is None:
            self._query = self._make("retrieval_query")
        return self._query
