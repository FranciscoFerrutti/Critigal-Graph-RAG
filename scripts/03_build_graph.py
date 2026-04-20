"""
Script 03 — Construcción del grafo en memoria y persistencia a parquet.

Lee:
  - config/graph_schema.yaml
  - data/acled_israel_2023.csv (o el path que se pase)

Escribe:
  - data/graph/nodes/{label}.parquet          (un archivo por tipo de nodo)
  - data/graph/relationships/{type}.parquet   (un archivo por tipo de relación)

No toca Neo4j. Sirve de "checkpoint" inspeccionable antes de cargar el grafo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.builder import GraphBuilder
from src.schema import GraphSchema


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construye nodos y relaciones del KG a parquet.")
    p.add_argument("--csv", default="data/acled_israel_2023.csv", help="CSV de entrada.")
    p.add_argument("--schema", default="config/graph_schema.yaml", help="Schema YAML.")
    p.add_argument("--out", default="data/graph", help="Carpeta de salida para los parquet.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    log.info("=== Build graph (03) — inicio ===")
    schema = GraphSchema.from_yaml(args.schema)
    log.info("Schema cargado: %d nodos, %d relaciones",
             len(schema.node_types), len(schema.relationship_types))

    df = pd.read_csv(args.csv, low_memory=False)
    log.info("CSV cargado: %d filas, %d columnas", len(df), len(df.columns))

    builder = GraphBuilder(schema)
    result = builder.build(df)
    log.info("Build OK\n%s", result.summary())

    out = Path(args.out)
    (out / "nodes").mkdir(parents=True, exist_ok=True)
    (out / "relationships").mkdir(parents=True, exist_ok=True)

    for label, ndf in result.nodes.items():
        path = out / "nodes" / f"{label}.parquet"
        ndf.to_parquet(path, index=False)
        log.info("Nodos %s -> %s (%d filas)", label, path, len(ndf))

    for rtype, rdf in result.relationships.items():
        path = out / "relationships" / f"{rtype}.parquet"
        rdf.to_parquet(path, index=False)
        log.info("Relaciones %s -> %s (%d filas)", rtype, path, len(rdf))

    log.info("=== Build graph (03) — fin ===")


if __name__ == "__main__":
    main()
