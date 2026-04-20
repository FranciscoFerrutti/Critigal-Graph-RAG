"""
Proveedor de embeddings local vía HuggingFace / sentence-transformers.

No requiere API key. Útil como fallback o para desarrollo offline.
Los modelos multilingües (ej: `paraphrase-multilingual-MiniLM-L12-v2`)
son los únicos con cross-lingual decente en local.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.embeddings import Embeddings

from src.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    HuggingFaceEmbeddings no distingue task_type entre documento y query,
    así que ambos getters devuelven la misma instancia.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._emb: Embeddings | None = None

    def _make(self) -> Embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.debug("Creando HuggingFaceEmbeddings: model=%s", self.model)
        return HuggingFaceEmbeddings(
            model_name=self.model,
            encode_kwargs={"batch_size": self.batch_size, "normalize_embeddings": True},
        )

    def get_document_embeddings(self) -> Embeddings:
        if self._emb is None:
            self._emb = self._make()
        return self._emb

    def get_query_embeddings(self) -> Embeddings:
        return self.get_document_embeddings()
