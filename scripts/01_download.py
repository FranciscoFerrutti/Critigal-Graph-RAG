"""
Script 01 — Descarga de datos ACLED.

Uso:
    python scripts/01_download.py
    python scripts/01_download.py --config config/dataset_filter.yaml

Requiere:
    - ACLED_API_KEY y ACLED_EMAIL en .env
    - config/dataset_filter.yaml con los filtros deseados
"""

import argparse
import logging
import sys
from pathlib import Path

# Permitir imports desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acled_client import ACLEDClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga datos de la API de ACLED")
    parser.add_argument(
        "--config",
        default="config/dataset_filter.yaml",
        help="Ruta al archivo de configuración de filtros (default: config/dataset_filter.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de salida del CSV (sobreescribe la del config si se especifica)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=== ACLED Download — Inicio ===")
    logger.info("Config: %s", args.config)

    try:
        client = ACLEDClient(config_path=args.config)
        df = client.download()

        if df.empty:
            logger.warning("No se descargaron registros. Verificá los filtros y credenciales.")
            sys.exit(1)

        saved_path = client.save(df, output_path=args.output)
        logger.info("=== Descarga completada: %d registros → %s ===", len(df), saved_path)

    except EnvironmentError as e:
        logger.error("Error de configuracion: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Error inesperado: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
