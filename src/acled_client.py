"""
Cliente para la API de ACLED (OAuth2).

Flujo de autenticación:
  1. POST https://acleddata.com/oauth/token con username + password
     → recibe access_token (Bearer, válido 24h)
  2. GET  https://acleddata.com/api/acled/read con Authorization: Bearer <token>

Lee los filtros de config/dataset_filter.yaml y descarga datos paginados,
guardando el resultado como CSV (o Parquet) en data/.
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_BASE_URL = "https://acleddata.com/api/acled/read"


class ACLEDClient:
    """Descarga eventos de la API de ACLED respetando los filtros del YAML."""

    def __init__(self, config_path: str = "config/dataset_filter.yaml") -> None:
        self.config = self._load_config(config_path)
        self._username = os.getenv("ACLED_USERNAME")
        self._password = os.getenv("ACLED_PASSWORD")
        self._token: str | None = None
        logger.debug("ACLEDClient inicializado con config: %s", self.config)
        logger.debug("Credenciales cargadas: username=%s, password=%s", bool(self._username), bool(self._password))
        self._validate_credentials()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _validate_credentials(self) -> None:
        if not self._username or not self._password:
            raise EnvironmentError(
                "Faltan credenciales ACLED. "
                "Definí ACLED_USERNAME y ACLED_PASSWORD en tu archivo .env"
            )

    # ------------------------------------------------------------------
    # Autenticación OAuth2
    # ------------------------------------------------------------------

    def _fetch_token(self) -> str:
        """Obtiene un Bearer token desde el endpoint OAuth2 de ACLED."""
        logger.debug("Obteniendo token ACLED desde %s", ACLED_TOKEN_URL)
        response = requests.post(
            ACLED_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "username": self._username,
                "password": self._password,
                "grant_type": "password",
                "client_id": "acled",           # valor fijo según documentación
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Error al obtener token ACLED: {response.status_code} {response.text}"
            )
        token = response.json().get("access_token")
        if not token:
            raise RuntimeError(f"Respuesta inesperada del token endpoint: {response.json()}")
        logger.info("Token ACLED obtenido correctamente (válido 24h)")
        return token

    @property
    def token(self) -> str:
        """Retorna el token, obteniéndolo si aún no existe."""
        if self._token is None:
            self._token = self._fetch_token()
        return self._token

    @property
    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Construcción de parámetros
    # ------------------------------------------------------------------

    def _build_date_range(self) -> tuple[str, str]:
        """Convierte la lista de años en un rango event_date BETWEEN."""
        years: list[int] = self.config["filters"]["year"]
        return f"{min(years)}-01-01", f"{max(years)}-12-31"

    def _build_country_param(self) -> str:
        """
        Construye el valor del parámetro 'country' según la sintaxis de ACLED.

        Un país:   "Colombia"
        Varios:    "Colombia:OR:country=Venezuela:OR:country=Ecuador"
        """
        countries: list[str] = self.config["filters"]["country"]
        if len(countries) == 1:
            return countries[0]
        first, *rest = countries
        return first + "".join(f":OR:country={c}" for c in rest)

    def _base_params(self) -> dict:
        start, end = self._build_date_range()
        params: dict = {
            "_format": "json",
            "country": self._build_country_param(),
            "event_date": f"{start}|{end}",
            "event_date_where": "BETWEEN",
        }

        event_types = self.config["filters"].get("event_type")
        if event_types:
            # Múltiples tipos se separan con "|"
            params["event_type"] = "|".join(event_types)

        return params

    # ------------------------------------------------------------------
    # Descarga
    # ------------------------------------------------------------------

    def _fetch_page(self, params: dict) -> list[dict]:
        """Hace un request autenticado y devuelve la lista de eventos."""
        safe_params = {k: v for k, v in params.items() if k != "_format"}
        logger.debug("GET %s params=%s", ACLED_BASE_URL, safe_params)

        response = requests.get(
            ACLED_BASE_URL,
            params=params,
            headers=self._auth_headers,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") != 200:
            raise RuntimeError(f"Error de API ACLED: {data.get('error', data)}")

        return data.get("data", [])

    def download(self) -> pd.DataFrame:
        """
        Descarga todos los eventos según los filtros configurados.

        Retorna un DataFrame con todos los registros.
        Si paginate=True en la config, itera por páginas hasta agotar resultados.
        """
        dl_cfg = self.config["download"]
        paginate: bool = dl_cfg.get("paginate", True)
        page_size: int = dl_cfg.get("page_size", 500)
        limit: int = dl_cfg.get("limit", 5000)

        all_records: list[dict] = []
        base = self._base_params()

        if not paginate:
            base["limit"] = limit
            records = self._fetch_page(base)
            all_records.extend(records)
            logger.info("%d registros descargados (sin paginacion)", len(records))
        else:
            page = 1
            downloaded = 0
            while downloaded < limit:
                batch_size = min(page_size, limit - downloaded)
                params = {**base, "limit": batch_size, "page": page}
                records = self._fetch_page(params)

                if not records:
                    logger.info("Pagina %d vacia — fin de resultados", page)
                    break

                all_records.extend(records)
                downloaded += len(records)
                logger.info(
                    "Pagina %d: %d registros (total acumulado: %d)",
                    page, len(records), downloaded,
                )

                if len(records) < batch_size:
                    break

                page += 1
                time.sleep(0.3)  # cortesía hacia la API

        df = pd.DataFrame(all_records)
        logger.info("Total de registros descargados: %d", len(df))
        return df

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, output_path: str | None = None) -> Path:
        """Guarda el DataFrame en la ruta indicada (CSV o Parquet)."""
        if output_path is None:
            output_path = self.config["download"]["output_path"]

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fmt = self.config["download"].get("output_format", "csv")
        if fmt == "parquet":
            df.to_parquet(path.with_suffix(".parquet"), index=False)
            saved = path.with_suffix(".parquet")
        else:
            df.to_csv(path, index=False, encoding="utf-8")
            saved = path

        logger.info("Datos guardados en: %s", saved)
        return saved

    def download_and_save(self) -> Path:
        """Atajo: descarga y guarda en un solo paso."""
        df = self.download()
        return self.save(df)
