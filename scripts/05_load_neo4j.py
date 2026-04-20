"""
Script 05 — Carga de nodos, relaciones y embeddings en Neo4j.

Lee:
  - data/graph/nodes/*.parquet
  - data/graph/relationships/*.parquet
  - data/graph/event_embeddings.parquet (opcional; si existe, adjunta vectores)
  - config/graph_schema.yaml
  - config/embeddings.yaml (para saber la dimensión del índice vectorial)

Modos:
  --mode incremental  (default): MERGE; re-ejecutable, idempotente.
  --mode destructive            : DETACH DELETE previo + recrear todo.

Orden de ejecución:
  1. (si destructive) drop_all
  2. create_constraints
  3. upsert_nodes (por label, siguiendo el schema)
  4. upsert_relationships (por type)
  5. attach_embeddings (si el parquet existe)
  6. create_vector_index
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_yaml, get_active_provider_config
from src.graph.neo4j_store import Neo4jStore
from src.neo4j_conn import get_driver
from src.schema import GraphSchema


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Carga el KG a Neo4j.")
    p.add_argument("--graph-dir", default="data/graph", help="Carpeta con nodes/ y relationships/")
    p.add_argument("--schema", default="config/graph_schema.yaml")
    p.add_argument("--embeddings-config", default="config/embeddings.yaml")
    p.add_argument("--mode", choices=["incremental", "destructive"], default="incremental",
                   help="incremental: MERGE. destructive: borra todo antes de cargar.")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="No subir embeddings aunque el parquet exista.")
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

    log.info("=== Load Neo4j (05) — inicio — mode=%s ===", args.mode)
    schema = GraphSchema.from_yaml(args.schema)
    emb_cfg = load_yaml(args.embeddings_config)
    active_emb = get_active_provider_config(emb_cfg)
    vector_dim = int(active_emb["dimensions"])

    graph_dir = Path(args.graph_dir)
    nodes_dir = graph_dir / "nodes"
    rels_dir = graph_dir / "relationships"
    emb_path = graph_dir / "event_embeddings.parquet"

    with get_driver() as driver:
        store = Neo4jStore(driver)

        if args.mode == "destructive":
            store.drop_all()

        store.create_constraints(schema)

        # 1) Nodos — orden: dependencias primero (EventType primary antes que sub es
        #    irrelevante porque MERGE es idempotente, pero mantener orden estable).
        for node_key, spec in schema.node_types.items():
            path = nodes_dir / f"{spec.label}.parquet"
            if not path.exists():
                log.warning("No hay parquet para %s, se salta", spec.label)
                continue
            df = pd.read_parquet(path)
            store.upsert_nodes(spec, df)

        # 2) Relaciones
        for rel_key, spec in schema.relationship_types.items():
            path = rels_dir / f"{spec.type}.parquet"
            if not path.exists():
                log.warning("No hay parquet para %s, se salta", spec.type)
                continue
            df = pd.read_parquet(path)
            store.upsert_relationships(spec, df, schema)

        # 3) Embeddings (si existen)
        if emb_path.exists() and not args.skip_embeddings:
            emb_df = pd.read_parquet(emb_path)
            store.attach_embeddings(
                node_label="Event",
                id_property="event_id",
                embedding_property=schema.vector_index.property,
                df=emb_df,
            )
        else:
            log.info("No se cargan embeddings (path=%s, skip=%s)", emb_path, args.skip_embeddings)

        # 4) Índice vectorial
        store.create_vector_index(schema, dimensions=vector_dim)

    log.info("=== Load Neo4j (05) — fin ===")


if __name__ == "__main__":
    main()
