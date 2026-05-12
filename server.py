"""
Servidor FastAPI para Critical Graph RAG chatbot.

Endpoints principales:
  - POST /chat: Procesa un mensaje de usuario y devuelve la respuesta del agente.
  - GET /health: Health check.

Configuración:
  - CORS habilitado para acceder desde cualquier origen.
  - Logging estructurado.
  - Auto-inicialización del agente en startup.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import init_agent, get_agent

# Cargar variables de entorno
load_dotenv(override=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request body para POST /chat."""

    message: str = Field(..., description="Pregunta del usuario en lenguaje natural (ES/EN).")
    top_k: int | None = Field(
        None,
        description="(Opcional) Cantidad de eventos a recuperar. Si no se especifica, usa el default del config.",
    )


class ChatResponse(BaseModel):
    """Response body de POST /chat."""

    response: str = Field(..., description="Respuesta del agente.")
    language: str = Field(..., description="Idioma detectado de la pregunta (es|en).")


class HealthResponse(BaseModel):
    """Response body de GET /health."""

    status: str = Field(..., description="Estado del servidor.")
    agent_ready: bool = Field(..., description="¿El agente está inicializado?")


# =============================================================================
# Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ciclo de vida de la aplicación FastAPI.
    - startup: Inicializar agente
    - shutdown: Cleanup (si es necesario)
    """
    # Startup
    logger.info("🚀 Iniciando servidor...")
    try:
        init_agent()
        logger.info("✓ Agente inicializado correctamente")
    except Exception as e:
        logger.error(f"✗ Error inicializando agente: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("🛑 Apagando servidor...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Critical Graph RAG API",
    description="Chatbot de análisis de conflictos e historia con Knowledge Graph + RAG",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios concretos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check del servidor y estado del agente."""
    try:
        agent = get_agent()
        agent_ready = agent is not None
    except Exception as e:
        logger.warning(f"Health check: agente no disponible - {e}")
        agent_ready = False

    return HealthResponse(
        status="ok",
        agent_ready=agent_ready,
    )


def _detect_language(text: str) -> str:
    """Detecta si un texto está en español o inglés."""
    es_words = {"qué", "por", "para", "como", "donde", "cuando", "el", "la", "los", "las"}
    en_words = {"what", "why", "how", "where", "when", "the", "a", "is", "are", "have"}

    text_lower = text.lower()
    es_score = sum(1 for w in es_words if w in text_lower)
    en_score = sum(1 for w in en_words if w in text_lower)

    return "es" if es_score > en_score else "en"


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Procesa un mensaje de usuario y devuelve la respuesta del agente.

    El agente decide qué tool usar (similarity_search, cypher_query, etc.)
    basándose en el contenido de la pregunta.

    Args:
        request: Una pregunta en lenguaje natural (ES/EN).

    Returns:
        ChatResponse con la respuesta del agente.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    try:
        # Obtener agente
        agent = get_agent()
        if agent is None:
            raise RuntimeError("Agente no inicializado")

        # Invocar agente
        logger.info(f"📨 Procesando mensaje: {request.message[:100]}...")
        response_text = agent.invoke(request.message)

        # Detectar idioma
        language = _detect_language(request.message)

        logger.info(f"✓ Respuesta generada (idioma: {language})")

        return ChatResponse(
            response=response_text,
            language=language,
        )

    except Exception as e:
        logger.error(f"Error procesando chat: {e}", exc_info=True)

        # Respuesta de error en el idioma detectado
        language = _detect_language(request.message)
        if language == "es":
            error_msg = f"Hubo un error procesando tu pregunta: {str(e)}"
        else:
            error_msg = f"There was an error processing your question: {str(e)}"

        raise HTTPException(status_code=500, detail=error_msg)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Por defecto, correr en localhost:8000
    # Cambiar host/port según necesidad
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("SERVER_PORT", "8000")))

    reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    logger.info(f"🚀 Iniciando servidor en {host}:{port}")
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
    )
