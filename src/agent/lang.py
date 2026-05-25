"""Detección de idioma (ES / EN) por heurística de stopwords."""

from __future__ import annotations

_ES_WORDS = {"qué", "por", "para", "como", "donde", "cuando", "el", "la", "los", "las"}
_EN_WORDS = {"what", "why", "how", "where", "when", "the", "a", "is", "are", "have"}


def detect_language(text: str) -> str:
    """Devuelve 'es' o 'en' según qué stopwords aparezcan más en el texto."""
    text_lower = text.lower()
    es_score = sum(1 for w in _ES_WORDS if w in text_lower)
    en_score = sum(1 for w in _EN_WORDS if w in text_lower)
    return "es" if es_score > en_score else "en"
