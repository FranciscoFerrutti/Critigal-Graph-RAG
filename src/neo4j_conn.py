"""
Fábrica de conexiones a Neo4j.

Soporta dos métodos de autenticación, en orden de prioridad:
  1. Usuario + contraseña  (NEO4J_USER + NEO4J_PASSWORD)
  2. Client credentials    (NEO4J_CLIENT_ID + NEO4J_CLIENT_SECRET)

Si ambos están configurados se usa el método 1 y se emite un warning.
Si ninguno está configurado se lanza EnvironmentError.
"""

import logging
import os

from dotenv import load_dotenv
from neo4j import Auth, GraphDatabase, Driver

# Forzar la carga del .env apenas se importa el módulo. override=True para que
# el .env gane sobre posibles env vars viejas del sistema (el mismo patrón
# que usamos en acled_client.py).
load_dotenv(override=True)

logger = logging.getLogger(__name__)


def _has(var: str) -> bool:
    return bool(os.getenv(var, "").strip())


def get_driver(uri: str | None = None) -> Driver:
    """
    Crea y devuelve un Driver de Neo4j autenticado.

    Parámetros
    ----------
    uri : str | None
        URI de conexión. Si es None se usa la variable de entorno NEO4J_URI.

    Returns
    -------
    neo4j.Driver — llamar a .close() cuando ya no se necesite,
                   o usar como context manager (with get_driver() as drv: ...).
    """
    neo4j_uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")

    has_basic = _has("NEO4J_USER") and _has("NEO4J_PASSWORD")
    has_oauth = _has("NEO4J_CLIENT_ID") and _has("NEO4J_CLIENT_SECRET")

    if has_basic and has_oauth:
        logger.warning(
            "Ambos métodos de autenticación están configurados "
            "(NEO4J_USER/PASSWORD y NEO4J_CLIENT_ID/SECRET). "
            "Se usará usuario + contraseña (prioridad 1). "
            "Para usar OAuth2 eliminá NEO4J_USER y NEO4J_PASSWORD del entorno."
        )

    if has_basic:
        auth = (os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
        logger.debug("Conectando a Neo4j con autenticación básica (usuario+contraseña)")
        return GraphDatabase.driver(neo4j_uri, auth=auth)

    if has_oauth:
        client_id = os.environ["NEO4J_CLIENT_ID"]
        client_secret = os.environ["NEO4J_CLIENT_SECRET"]
        # Neo4j 5.x soporta Bearer token via Auth("bearer", token).
        # Para client_credentials el token se obtiene del IdP primero;
        # aquí se asume que CLIENT_SECRET es directamente el bearer token
        # (caso AuraDB / Neo4j Aura con SSO habilitado).
        # Si tu IdP requiere un intercambio OAuth2 previo, obtené el token
        # allí y pasalo como NEO4J_CLIENT_SECRET.
        auth = Auth(scheme="bearer", credentials=client_secret, principal=client_id)
        logger.debug("Conectando a Neo4j con autenticación OAuth2 (client credentials)")
        return GraphDatabase.driver(neo4j_uri, auth=auth)

    raise EnvironmentError(
        "No se encontraron credenciales de Neo4j. "
        "Definí NEO4J_USER + NEO4J_PASSWORD, "
        "o NEO4J_CLIENT_ID + NEO4J_CLIENT_SECRET en tu archivo .env."
    )
