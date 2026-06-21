"""Entrypoint serverless para Vercel.

Vercel agrega el Root Directory (web/) al sys.path, así que importamos la app
FastAPI definida en web/app.py. Vercel detecta la variable ASGI `app`.
"""

from app import app  # noqa: F401  (re-exporta la app para el runtime de Vercel)
