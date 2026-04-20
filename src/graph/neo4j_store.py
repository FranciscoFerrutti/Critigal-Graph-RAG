"""
Interfaz `GraphStore` + implementación Neo4j para cargar el KG.

Expone:
  - create_constraints(): unicidad por nodo según el schema
  - create_vector_index(): índice vectorial sobre Event.notes
  - upsert_nodes(): MERGE por dedup keys, SET de propiedades
  - upsert_relationships(): MERGE entre nodos identificados por dedup keys
  - attach_embeddings(): escribe embeddings precomputados en Event
  - drop_all(): DETACH DELETE de todo el grafo (para --mode destructive)

Diseño:
  - La interfaz `GraphStore` permite a futuro swap por otro store (Memgraph,
    FalkorDB, etc.) sin tocar los scripts.
  - Las queries Cypher se construyen desde el schema (NodeSpec / RelationshipSpec)
    para mantener la consistencia: cambiar un dedup_key o un atributo en el YAML
    se refleja automáticamente en las queries de carga.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable

import pandas as pd
from neo4j import Driver

from src.schema import GraphSchema, NodeSpec, RelationshipSpec

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Interfaz
# -----------------------------------------------------------------------------

class GraphStore(ABC):
    """Contrato de cualquier backend de grafos."""

    @abstractmethod
    def drop_all(self) -> None: ...

    @abstractmethod
    def create_constraints(self, schema: GraphSchema) -> None: ...

    @abstractmethod
    def create_vector_index(self, schema: GraphSchema, dimensions: int) -> None: ...

    @abstractmethod
    def upsert_nodes(self, spec: NodeSpec, df: pd.DataFrame, batch_size: int = 1000) -> int: ...

    @abstractmethod
    def upsert_relationships(
        self,
        spec: RelationshipSpec,
        df: pd.DataFrame,
        schema: GraphSchema,
        batch_size: int = 1000,
    ) -> int: ...

    @abstractmethod
    def attach_embeddings(
        self,
        node_label: str,
        id_property: str,
        embedding_property: str,
        df: pd.DataFrame,
        batch_size: int = 500,
    ) -> int: ...


# -----------------------------------------------------------------------------
# Implementación Neo4j
# -----------------------------------------------------------------------------

class Neo4jStore(GraphStore):
    def __init__(self, driver: Driver) -> None:
        self.driver = driver

    # --- administrativas ----------------------------------------------------

    def drop_all(self) -> None:
        logger.warning("DROP ALL: borrando todos los nodos, relaciones e índices del grafo.")
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
            # Orden importa: primero los constraints, después los índices.
            # Los índices que respaldan un constraint no se pueden dropear sueltos
            # (Neo4j error: Index belongs to constraint).
            for row in s.run("SHOW CONSTRAINTS YIELD name").data():
                s.run(f"DROP CONSTRAINT {row['name']} IF EXISTS")
            for row in s.run("SHOW INDEXES YIELD name").data():
                s.run(f"DROP INDEX {row['name']} IF EXISTS")

    def create_constraints(self, schema: GraphSchema) -> None:
        """Unicidad por dedup_keys de cada nodo."""
        with self.driver.session() as s:
            for spec in schema.node_types.values():
                keys = spec.dedup_keys
                if not keys:
                    continue
                if len(keys) == 1:
                    cypher = (
                        f"CREATE CONSTRAINT {spec.label.lower()}_unique IF NOT EXISTS "
                        f"FOR (n:{spec.label}) REQUIRE n.{keys[0]} IS UNIQUE"
                    )
                else:
                    props = ", ".join(f"n.{k}" for k in keys)
                    cypher = (
                        f"CREATE CONSTRAINT {spec.label.lower()}_unique IF NOT EXISTS "
                        f"FOR (n:{spec.label}) REQUIRE ({props}) IS UNIQUE"
                    )
                logger.info("Constraint: %s", cypher)
                s.run(cypher)

    def create_vector_index(self, schema: GraphSchema, dimensions: int) -> None:
        vi = schema.vector_index
        cypher = f"""
        CREATE VECTOR INDEX {vi.name} IF NOT EXISTS
        FOR (n:{vi.node_label}) ON (n.{vi.property})
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: '{vi.similarity_function}'
        }}}}
        """
        logger.info("Vector index: %s (dims=%d)", vi.name, dimensions)
        with self.driver.session() as s:
            s.run(cypher)

    # --- upserts ------------------------------------------------------------

    def upsert_nodes(
        self, spec: NodeSpec, df: pd.DataFrame, batch_size: int = 1000
    ) -> int:
        if df.empty:
            return 0

        keys = spec.dedup_keys
        set_props = [c for c in df.columns if c not in keys]

        merge_keys = ", ".join(f"{k}: row.{k}" for k in keys)
        set_clause = ", ".join(f"n.{p} = row.{p}" for p in set_props)
        cypher = (
            f"UNWIND $batch AS row "
            f"MERGE (n:{spec.label} {{ {merge_keys} }}) "
        )
        if set_clause:
            cypher += f"SET {set_clause}"

        total = 0
        with self.driver.session() as s:
            for chunk in _chunked(df, batch_size):
                batch = _records_for_cypher(chunk)
                s.run(cypher, batch=batch)
                total += len(batch)
        logger.info("Upsert %s: %d nodos", spec.label, total)
        return total

    def upsert_relationships(
        self,
        spec: RelationshipSpec,
        df: pd.DataFrame,
        schema: GraphSchema,
        batch_size: int = 1000,
    ) -> int:
        if df.empty:
            return 0

        from_spec = schema.node(spec.from_node)
        to_spec = schema.node(spec.to_node)

        from_match, from_params = _build_node_match("a", from_spec, df, side="from")
        to_match, to_params = _build_node_match("b", to_spec, df, side="to")
        rel_props = [a.name for a in spec.attributes]
        set_clause = (
            "SET " + ", ".join(f"r.{p} = row.{p}" for p in rel_props) if rel_props else ""
        )

        cypher = (
            f"UNWIND $batch AS row "
            f"{from_match} "
            f"{to_match} "
            f"MERGE (a)-[r:{spec.type}]->(b) "
            f"{set_clause}"
        )

        total = 0
        needed_cols = set(from_params + to_params + rel_props)
        with self.driver.session() as s:
            for chunk in _chunked(df, batch_size):
                batch = _records_for_cypher(chunk[list(needed_cols & set(chunk.columns))])
                s.run(cypher, batch=batch)
                total += len(batch)
        logger.info("Upsert %s: %d relaciones", spec.type, total)
        return total

    def attach_embeddings(
        self,
        node_label: str,
        id_property: str,
        embedding_property: str,
        df: pd.DataFrame,
        batch_size: int = 500,
    ) -> int:
        if df.empty:
            return 0
        cypher = (
            f"UNWIND $batch AS row "
            f"MATCH (n:{node_label} {{ {id_property}: row.id }}) "
            f"CALL db.create.setNodeVectorProperty(n, '{embedding_property}', row.embedding)"
        )
        total = 0
        with self.driver.session() as s:
            for chunk in _chunked(df, batch_size):
                batch = [
                    {"id": r[id_property], "embedding": list(r["embedding"])}
                    for r in chunk.to_dict(orient="records")
                ]
                s.run(cypher, batch=batch)
                total += len(batch)
        logger.info("Attach embeddings (%s.%s): %d nodos", node_label, embedding_property, total)
        return total


# -----------------------------------------------------------------------------
# Utilidades internas
# -----------------------------------------------------------------------------

def _chunked(df: pd.DataFrame, size: int) -> Iterable[pd.DataFrame]:
    for i in range(0, len(df), size):
        yield df.iloc[i : i + size]


def _records_for_cypher(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convierte a list[dict] y reemplaza NaN por None (Neo4j no acepta NaN)."""
    records = df.to_dict(orient="records")
    return [{k: (None if _is_nanlike(v) else v) for k, v in r.items()} for r in records]


def _is_nanlike(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:  # NaN
        return True
    return False


def _build_node_match(
    alias: str, spec: NodeSpec, df: pd.DataFrame, side: str
) -> tuple[str, list[str]]:
    """
    Genera un `MATCH (alias:Label {props})` usando las dedup_keys del nodo.
    Las columnas del DataFrame deben venir con el prefijo `{side}_`.
    """
    prefix = f"{side}_"

    # Mapeo dedup_key -> nombre de columna en el DF
    col_for_key: dict[str, str] = {}
    for key in spec.dedup_keys:
        # Aceptar dos convenciones: "from_<key>" o "from_<snake(label)>_<key>"
        candidates = [
            f"{prefix}{key}",
            f"{prefix}{_snake(spec.label)}_{key}",
        ]
        match = next((c for c in candidates if c in df.columns), None)
        if match is None:
            raise ValueError(
                f"Relación → nodo {spec.label}: falta columna en DF para key '{key}'. "
                f"Buscamos alguna de {candidates}; columnas presentes: {list(df.columns)}"
            )
        col_for_key[key] = match

    props = ", ".join(f"{k}: row.{col_for_key[k]}" for k in spec.dedup_keys)
    cypher = f"MATCH ({alias}:{spec.label} {{ {props} }})"
    return cypher, list(col_for_key.values())


def _snake(label: str) -> str:
    out = []
    for i, ch in enumerate(label):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
