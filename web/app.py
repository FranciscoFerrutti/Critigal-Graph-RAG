"""Capa web de CriticalGraphRAG: UI de chatbot + proxy hacia el agente en producción.

El agente ya corre como servicio remoto (Render) y expone POST /chat con el
contrato {"message": str, "top_k": int | null} -> {"response": str, "language": str}.
El endpoint remoto es stateless: no acepta session_id ni historial, por lo que
la memoria de conversación vive acá, en un dict en memoria por session_id.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv(override=True)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

AGENT_CHAT_URL = os.getenv(
    "AGENT_CHAT_URL", "https://critigal-graph-rag.onrender.com/chat"
)
# Render (free tier) puede tardar ~1 min en despertar de un cold start.
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="CriticalGraphRAG Web",
    description="UI de chatbot sobre el agente ACLED Israel/Palestina.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memoria de conversación: session_id -> lista de mensajes {role, content}.
# Sin persistencia: se pierde al reiniciar el servidor (comportamiento esperado).
_sessions: dict[str, list[dict[str, str]]] = defaultdict(list)


class ChatRequest(BaseModel):
    """Request del frontend hacia este backend."""

    message: str = Field(..., min_length=1, description="Pregunta del usuario.")
    session_id: str = Field(..., min_length=1, description="UUID generado por el frontend.")
    top_k: int | None = Field(
        default=None, description="(Opcional) Cantidad de eventos a recuperar."
    )


class ChatResponse(BaseModel):
    """Response hacia el frontend."""

    response: str
    language: str
    session_id: str


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "sessions": len(_sessions)}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Proxy hacia el agente en producción, con memoria local por sesión.

    El agente remoto no acepta historial, así que solo se reenvía el último
    mensaje; el historial completo queda registrado en _sessions.
    """
    history = _sessions[request.session_id]
    history.append({"role": "user", "content": request.message})

    payload: dict = {"message": request.message}
    if request.top_k is not None:
        payload["top_k"] = request.top_k

    try:
        upstream = requests.post(
            AGENT_CHAT_URL, json=payload, timeout=AGENT_TIMEOUT_SECONDS
        )
        upstream.raise_for_status()
        data = upstream.json()
    except requests.Timeout:
        logger.exception("Timeout consultando al agente")
        raise HTTPException(
            status_code=504,
            detail="El agente tardó demasiado en responder. Probá de nuevo en unos segundos.",
        )
    except requests.RequestException:
        logger.exception("Error consultando al agente")
        raise HTTPException(
            status_code=502,
            detail="No se pudo contactar al agente en producción.",
        )

    answer = data.get("response", "")
    history.append({"role": "assistant", "content": answer})

    return ChatResponse(
        response=answer,
        language=data.get("language", "es"),
        session_id=request.session_id,
    )


@app.get("/history/{session_id}")
def get_history(session_id: str) -> list[dict[str, str]]:
    """Devuelve el historial en memoria de una sesión (vacío si no existe)."""
    return _sessions.get(session_id, [])


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
