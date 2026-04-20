"""
Proveedores de embeddings abstraídos detrás de una interfaz común.

Uso típico:

    from src.config import load_yaml
    from src.embeddings import get_embedding_provider

    cfg = load_yaml("config/embeddings.yaml")
    provider = get_embedding_provider(cfg)
    doc_embedder = provider.get_document_embeddings()
    query_embedder = provider.get_query_embeddings()

Cambiar de proveedor = editar `provider:` en embeddings.yaml.
"""

from src.embeddings.base import EmbeddingProvider, get_embedding_provider

__all__ = ["EmbeddingProvider", "get_embedding_provider"]
