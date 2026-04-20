"""
Parseo y validación de `config/graph_schema.yaml`.

El schema es el contrato central del pipeline de construcción del KG.
Este módulo expone dataclasses tipadas y un `GraphSchema.from_yaml(path)`
que lo carga y valida estructuralmente. Cualquier script que necesite
conocer nodos/relaciones entra por acá, nunca leyendo el YAML directo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class AttributeSpec:
    """Describe una propiedad de un nodo o relación."""
    name: str
    type: str
    source_column: str | list[str] | None = None
    derived_from: str | None = None       # nombre de otro atributo del que deriva
    part: str | None = None               # 'year' | 'month' | 'day' (para fechas)
    format: str | None = None             # ej: 'YYYY-MM'
    formula: str | None = None            # expresión libre (ej: "1 + count('/')")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AttributeSpec":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Claves desconocidas en attribute {d.get('name')}: {unknown}")
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class NodeSpec:
    """Definición estructural de un tipo de nodo."""
    key: str                              # key bajo `node_types:` (ej: "Event")
    label: str                            # label en Neo4j
    source_id_column: str | None
    deduplicate_on: str | list[str] | None
    extraction: Any                       # dict o list; parseado según el tipo
    attributes: list[AttributeSpec] = field(default_factory=list)
    text_for_embedding: list[str] | None = None

    @property
    def id_strategy(self) -> str:
        """'source_id' si dedup por columna ID, 'dedup' si por valor."""
        if self.source_id_column:
            return "source_id"
        if self.deduplicate_on:
            return "dedup"
        raise ValueError(f"Nodo {self.key}: debe declarar source_id_column o deduplicate_on")

    @property
    def dedup_keys(self) -> list[str]:
        """Lista de atributos que definen identidad (para MERGE en Neo4j)."""
        if self.source_id_column:
            # por convención la propiedad se llama igual que el atributo mapeado
            for attr in self.attributes:
                if attr.source_column == self.source_id_column:
                    return [attr.name]
            raise ValueError(
                f"Nodo {self.key}: source_id_column='{self.source_id_column}' "
                f"no aparece en ningún atributo"
            )
        if isinstance(self.deduplicate_on, str):
            return [self.deduplicate_on]
        return list(self.deduplicate_on or [])

    @classmethod
    def from_dict(cls, key: str, d: dict[str, Any]) -> "NodeSpec":
        attrs = [AttributeSpec.from_dict(a) for a in d.get("attributes", [])]
        return cls(
            key=key,
            label=d.get("label", key),
            source_id_column=d.get("source_id_column"),
            deduplicate_on=d.get("deduplicate_on"),
            extraction=d.get("extraction", {}),
            attributes=attrs,
            text_for_embedding=d.get("text_for_embedding"),
        )


@dataclass
class RelationshipSpec:
    """Definición estructural de un tipo de relación."""
    key: str
    type: str                             # nombre de la relación en Neo4j
    from_node: str                        # key del nodo origen
    to_node: str                          # key del nodo destino
    extraction: Any
    attributes: list[AttributeSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, key: str, d: dict[str, Any]) -> "RelationshipSpec":
        attrs = [AttributeSpec.from_dict(a) for a in d.get("attributes", [])]
        return cls(
            key=key,
            type=d.get("type", key),
            from_node=d["from_node"],
            to_node=d["to_node"],
            extraction=d.get("extraction", {}),
            attributes=attrs,
        )


@dataclass
class VectorIndexSpec:
    """Config del índice vectorial único de Event.notes."""
    name: str
    node_label: str
    property: str
    similarity_function: str              # 'cosine' | 'euclidean'
    dimensions: int | None = None         # se inyecta desde embeddings.yaml

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VectorIndexSpec":
        return cls(
            name=d["name"],
            node_label=d["node_label"],
            property=d["property"],
            similarity_function=d["similarity_function"],
            dimensions=d.get("dimensions"),
        )


@dataclass
class GraphSchema:
    """Raíz del schema: nodos + relaciones + config Neo4j."""
    node_types: dict[str, NodeSpec]
    relationship_types: dict[str, RelationshipSpec]
    vector_index: VectorIndexSpec

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GraphSchema":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        nodes = {
            k: NodeSpec.from_dict(k, v) for k, v in raw.get("node_types", {}).items()
        }
        rels = {
            k: RelationshipSpec.from_dict(k, v)
            for k, v in raw.get("relationship_types", {}).items()
        }
        vi = VectorIndexSpec.from_dict(raw["neo4j"]["vector_index"])
        schema = cls(node_types=nodes, relationship_types=rels, vector_index=vi)
        schema.validate()
        return schema

    def validate(self) -> None:
        """Checks de consistencia que fallan temprano."""
        node_keys = set(self.node_types)
        for rel in self.relationship_types.values():
            if rel.from_node not in node_keys:
                raise ValueError(
                    f"Relación '{rel.key}': from_node='{rel.from_node}' no existe"
                )
            if rel.to_node not in node_keys:
                raise ValueError(
                    f"Relación '{rel.key}': to_node='{rel.to_node}' no existe"
                )

        for node in self.node_types.values():
            if node.source_id_column and node.deduplicate_on:
                raise ValueError(
                    f"Nodo '{node.key}': source_id_column y deduplicate_on "
                    f"son mutuamente excluyentes"
                )
            # disparar id_strategy para validar consistencia
            _ = node.id_strategy

    # --------------------------------------------------------------
    # Helpers de consulta
    # --------------------------------------------------------------

    def node(self, key: str) -> NodeSpec:
        return self.node_types[key]

    def relationship(self, key: str) -> RelationshipSpec:
        return self.relationship_types[key]

    def nodes_requiring_embedding(self) -> list[NodeSpec]:
        return [n for n in self.node_types.values() if n.text_for_embedding]
