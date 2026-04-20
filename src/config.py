"""
Carga y resolución de configuraciones YAML del proyecto.

Centraliza el acceso a:
  - config/graph_schema.yaml
  - config/embeddings.yaml
  - config/agent.yaml
  - config/cypher_library.yaml
  - config/dataset_filter.yaml

Y a las API keys referenciadas por `api_key_env` dentro de los YAML.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path("config")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Lee un YAML y lo devuelve como dict. Lanza FileNotFoundError si no existe."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config no encontrado: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_api_key(api_key_env: str | None) -> str | None:
    """
    Resuelve una API key a partir del nombre de variable de entorno declarado
    en el YAML. Si `api_key_env` es None devuelve None (proveedores locales).
    Si está definido pero la env var no existe, lanza EnvironmentError.
    """
    if api_key_env is None:
        return None
    value = os.getenv(api_key_env, "").strip()
    if not value:
        raise EnvironmentError(
            f"Falta la variable de entorno '{api_key_env}' referenciada "
            f"por el YAML de configuración. Definila en .env."
        )
    return value


def get_active_provider_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Dado un YAML con la forma:
        provider: "google"
        google: { ... }
        huggingface: { ... }
    devuelve el sub-dict del proveedor activo con la key `provider` agregada.
    """
    provider = config.get("provider")
    if not provider:
        raise ValueError("El config no tiene campo 'provider'")
    if provider not in config:
        raise ValueError(
            f"El config declara provider='{provider}' pero no tiene sección '{provider}'"
        )
    active = dict(config[provider])
    active["provider"] = provider
    return active
