"""
Transforma un DataFrame ACLED en tablas de nodos y relaciones listas para Neo4j.

Entrada:  DataFrame del CSV + GraphSchema.
Salida:   (nodes_by_label, relationships_by_type), ambos dict[str, DataFrame].

La lógica dispatchea según las keys de `extraction` del schema:
  - one_per_row:         un nodo por fila del CSV
  - source_column:       extrae una sola columna (con split opcional)
  - source_columns:      extrae varias columnas (lista plana o lista de dicts)
  - pairs:               para relaciones que emparejan dos columnas
  - etc.

Los atributos derivados (`derived_from` + `part` o `format`, y `formula`) se
resuelven acá para que los parquets resultantes ya tengan el shape final.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

from src.schema import AttributeSpec, GraphSchema, NodeSpec, RelationshipSpec

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Resultado del builder: nodos y relaciones agrupados por label/type."""
    nodes: dict[str, pd.DataFrame]
    relationships: dict[str, pd.DataFrame]

    def summary(self) -> str:
        node_s = ", ".join(f"{k}={len(v)}" for k, v in self.nodes.items())
        rel_s = ", ".join(f"{k}={len(v)}" for k, v in self.relationships.items())
        return f"Nodes: {node_s}\nRelationships: {rel_s}"


# -----------------------------------------------------------------------------
# Helpers puros
# -----------------------------------------------------------------------------

def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _maybe_split(value, split_on: str | None, strip: bool) -> list[str]:
    """Devuelve la lista de valores no vacíos tras aplicar (o no) split."""
    if _is_missing(value):
        return []
    text = str(value)
    parts = text.split(split_on) if split_on else [text]
    if strip:
        parts = [p.strip() for p in parts]
    return [p for p in parts if p]


def _derive_from_date(series: pd.Series, part: str | None, fmt: str | None) -> pd.Series:
    """Extrae un componente de una fecha (o la formatea)."""
    dt = pd.to_datetime(series, errors="coerce")
    if part == "year":
        return dt.dt.year.astype("Int64")
    if part == "month":
        return dt.dt.month.astype("Int64")
    if part == "day":
        return dt.dt.day.astype("Int64")
    if fmt == "YYYY-MM":
        return dt.dt.strftime("%Y-%m")
    raise ValueError(f"Derivación de fecha no soportada: part={part!r} format={fmt!r}")


def _apply_formula(series: pd.Series, formula: str) -> pd.Series:
    """
    Resuelve la fórmula declarada en el schema. Por ahora soporta
    `1 + count('/')` que es la única usada (specificity_level).
    Ampliable con un parser si aparecen más casos.
    """
    f = formula.replace(" ", "")
    if f == "1+count('/')" or f == '1+count("/")':
        return series.fillna("").astype(str).str.count(re.escape("/")).add(1).astype(int)
    raise NotImplementedError(f"Formula no soportada: {formula!r}")


# -----------------------------------------------------------------------------
# Builder principal
# -----------------------------------------------------------------------------

class GraphBuilder:
    """Construye nodos y relaciones a partir del DataFrame ACLED + schema."""

    def __init__(self, schema: GraphSchema) -> None:
        self.schema = schema

    # ------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> BuildResult:
        logger.info("Build iniciado: %d filas del CSV", len(df))
        nodes: dict[str, pd.DataFrame] = {}
        for key, spec in self.schema.node_types.items():
            nodes[spec.label] = self._build_node(spec, df)
            logger.info("Nodo %s: %d filas", spec.label, len(nodes[spec.label]))

        rels: dict[str, pd.DataFrame] = {}
        for key, spec in self.schema.relationship_types.items():
            rels[spec.type] = self._build_relationship(spec, df)
            logger.info("Relación %s: %d filas", spec.type, len(rels[spec.type]))

        return BuildResult(nodes=nodes, relationships=rels)

    # ------------------------------------------------------------
    # Nodos
    # ------------------------------------------------------------

    def _build_node(self, spec: NodeSpec, df: pd.DataFrame) -> pd.DataFrame:
        extraction = spec.extraction
        # 1) una fila del CSV = un nodo (Event, Location, Source, ...)
        if isinstance(extraction, dict) and extraction.get("one_per_row"):
            return self._node_one_per_row(spec, df, extraction)

        # 2) EventType: lista de extracciones (primary + sub)
        if isinstance(extraction, list):
            return self._node_from_multi_extractions(spec, df, extraction)

        # 3) DisorderType / Month: dict con source_column (opcional split)
        if isinstance(extraction, dict) and "source_column" in extraction:
            return self._node_from_single_column(spec, df, extraction)

        # 4) Actor / ActorType: dict con source_columns (lista, con o sin split por item)
        if isinstance(extraction, dict) and "source_columns" in extraction:
            return self._node_from_many_columns(spec, df, extraction)

        raise ValueError(f"No sé cómo extraer el nodo {spec.key}: extraction={extraction}")

    def _node_one_per_row(
        self, spec: NodeSpec, df: pd.DataFrame, extraction: dict
    ) -> pd.DataFrame:
        """Cada fila del CSV aporta un nodo. Se usa para Event, Location, Source."""
        cols: dict[str, pd.Series] = {}
        for attr in spec.attributes:
            cols[attr.name] = self._materialize_attr(attr, df)

        out = pd.DataFrame(cols)

        # Source: la columna 'source' puede venir con ";"
        split_on = extraction.get("source_split_on")
        if split_on:
            strip = extraction.get("source_strip", True)
            # explode de la propia columna 'source' (el atributo se llama igual)
            target = "source"
            out[target] = out[target].apply(lambda v: _maybe_split(v, split_on, strip))
            out = out.explode(target, ignore_index=True)
            out = out.dropna(subset=[target])

        out = self._dedup(out, spec)
        return out

    def _node_from_single_column(
        self, spec: NodeSpec, df: pd.DataFrame, extraction: dict
    ) -> pd.DataFrame:
        """Extrae un nodo por valor único de una columna (con split opcional)."""
        column = extraction["source_column"]
        split_on = extraction.get("split_on")
        strip = extraction.get("strip", True)

        values = []
        for raw in df[column]:
            values.extend(_maybe_split(raw, split_on, strip))

        # Month es un caso especial: deriva todos sus atributos de la fecha
        if spec.key == "Month":
            return self._build_month(spec, df[column])

        # DisorderType y similares: un solo atributo (name)
        name_attr = next(a for a in spec.attributes if a.name in spec.dedup_keys)
        out = pd.DataFrame({name_attr.name: values})
        out = self._dedup(out, spec)
        # atributos derivados (ej: specificity_level)
        self._apply_derived(spec, out)
        return out

    def _build_month(self, spec: NodeSpec, date_series: pd.Series) -> pd.DataFrame:
        """Deriva month_id, year, month desde event_date."""
        dt = pd.to_datetime(date_series, errors="coerce").dropna()
        df = pd.DataFrame({
            "month_id": dt.dt.strftime("%Y-%m"),
            "year": dt.dt.year.astype("Int64"),
            "month": dt.dt.month.astype("Int64"),
        })
        df = df.drop_duplicates(subset=["month_id"]).reset_index(drop=True)
        return df

    def _node_from_many_columns(
        self, spec: NodeSpec, df: pd.DataFrame, extraction: dict
    ) -> pd.DataFrame:
        """
        Actor / ActorType: recolectar valores de varias columnas y deduplicar.
        Soporta item como string (columna sola) o dict (con split_on/strip).
        """
        values: list[str] = []
        for item in extraction["source_columns"]:
            if isinstance(item, str):
                column, split_on, strip = item, None, True
            else:
                column = item["column"]
                split_on = item.get("split_on")
                strip = item.get("strip", True)
            for raw in df[column]:
                values.extend(_maybe_split(raw, split_on, strip))

        name_attr = next(a for a in spec.attributes if a.name in spec.dedup_keys)
        out = pd.DataFrame({name_attr.name: values})
        out = self._dedup(out, spec)
        self._apply_derived(spec, out)
        return out

    def _node_from_multi_extractions(
        self, spec: NodeSpec, df: pd.DataFrame, extractions: list[dict]
    ) -> pd.DataFrame:
        """EventType: varias extracciones que añaden un atributo 'level'."""
        parts = []
        for ext in extractions:
            column = ext["source_column"]
            level = ext.get("level")
            sub = pd.DataFrame({"name": df[column].dropna().astype(str).str.strip()})
            sub = sub[sub["name"] != ""]
            if level:
                sub["level"] = level
            parts.append(sub)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        out = self._dedup(out, spec)
        self._apply_derived(spec, out)
        return out

    # ------------------------------------------------------------
    # Atributos
    # ------------------------------------------------------------

    def _materialize_attr(self, attr: AttributeSpec, df: pd.DataFrame) -> pd.Series:
        """Convierte la definición de un atributo en una Series sobre el CSV."""
        if attr.derived_from:
            return _derive_from_date(df[attr.derived_from], attr.part, attr.format)
        if attr.source_column and isinstance(attr.source_column, str):
            return df[attr.source_column]
        # fallback: columna vacía (se rellena después si aplica)
        return pd.Series([None] * len(df))

    def _apply_derived(self, spec: NodeSpec, out: pd.DataFrame) -> None:
        """Calcula atributos derivados sobre un DataFrame ya deduplicado."""
        for attr in spec.attributes:
            if attr.formula and attr.derived_from:
                if attr.derived_from not in out.columns:
                    raise ValueError(
                        f"{spec.key}.{attr.name} deriva de '{attr.derived_from}' "
                        f"pero esa columna no está presente todavía"
                    )
                out[attr.name] = _apply_formula(out[attr.derived_from], attr.formula)

    def _dedup(self, df: pd.DataFrame, spec: NodeSpec) -> pd.DataFrame:
        """Deduplica por las columnas de identidad del nodo."""
        keys = spec.dedup_keys
        missing = [k for k in keys if k not in df.columns]
        if missing:
            raise ValueError(f"Nodo {spec.key}: faltan columnas de dedup {missing}")
        # dropna sobre las keys para no arrastrar identidades vacías
        df = df.dropna(subset=keys).drop_duplicates(subset=keys).reset_index(drop=True)
        return df

    # ------------------------------------------------------------
    # Relaciones
    # ------------------------------------------------------------

    def _build_relationship(
        self, spec: RelationshipSpec, df: pd.DataFrame
    ) -> pd.DataFrame:
        # Tabla base con el event_id para poder unir luego en Neo4j.
        event_id_col = self.schema.node("Event").source_id_column
        if event_id_col is None:
            raise ValueError("Event debe tener source_id_column (event_id_cnty).")

        extraction = spec.extraction

        if spec.key == "SUBTYPE_OF":
            return self._rel_subtype_of(spec, df, extraction)
        if spec.key == "INVOLVED_IN":
            return self._rel_involved_in(spec, df, extraction, event_id_col)
        if spec.key == "HAS_TYPE":
            return self._rel_has_type(spec, df, extraction)
        if spec.key == "OF_TYPE":
            return self._rel_of_type(spec, df, event_id_col)
        if spec.key == "IN_MONTH":
            return self._rel_in_month(spec, df, event_id_col)
        if spec.key == "AT_LOCATION":
            return self._rel_at_location(spec, df, event_id_col)
        if spec.key == "REPORTED_BY":
            return self._rel_reported_by(spec, df, event_id_col)
        if spec.key == "OF_DISORDER":
            return self._rel_of_disorder(spec, df, event_id_col)

        raise NotImplementedError(f"Relación no soportada: {spec.key}")

    def _rel_of_type(self, spec, df, event_id_col) -> pd.DataFrame:
        col = spec.extraction["source_column"]
        out = pd.DataFrame({
            "from_event_id": df[event_id_col],
            "to_event_type_name": df[col].astype(str).str.strip(),
        })
        return out.dropna().query("to_event_type_name != ''").reset_index(drop=True)

    def _rel_subtype_of(self, spec, df, extraction) -> pd.DataFrame:
        child_col = extraction["child_column"]
        parent_col = extraction["parent_column"]
        pairs = df[[child_col, parent_col]].dropna().drop_duplicates()
        return pairs.rename(
            columns={child_col: "from_event_type_name", parent_col: "to_event_type_name"}
        ).reset_index(drop=True)

    def _rel_in_month(self, spec, df, event_id_col) -> pd.DataFrame:
        dt = pd.to_datetime(df[spec.extraction["source_column"]], errors="coerce")
        out = pd.DataFrame({
            "from_event_id": df[event_id_col],
            "to_month_id": dt.dt.strftime("%Y-%m"),
        })
        return out.dropna().reset_index(drop=True)

    def _rel_at_location(self, spec, df, event_id_col) -> pd.DataFrame:
        out = pd.DataFrame({
            "from_event_id": df[event_id_col],
            "to_latitude": df["latitude"],
            "to_longitude": df["longitude"],
        })
        return out.dropna(subset=["to_latitude", "to_longitude"]).reset_index(drop=True)

    def _rel_reported_by(self, spec, df, event_id_col) -> pd.DataFrame:
        ext = spec.extraction
        col = ext["source_column"]
        split_on = ext.get("split_on")
        strip = ext.get("strip", True)
        tmp = df[[event_id_col, col, "source_scale"]].rename(
            columns={event_id_col: "from_event_id", col: "to_source", "source_scale": "to_source_scale"}
        )
        tmp["to_source"] = tmp["to_source"].apply(lambda v: _maybe_split(v, split_on, strip))
        tmp = tmp.explode("to_source", ignore_index=True)
        tmp = tmp.dropna(subset=["to_source", "to_source_scale"]).reset_index(drop=True)
        return tmp

    def _rel_of_disorder(self, spec, df, event_id_col) -> pd.DataFrame:
        ext = spec.extraction
        col = ext["source_column"]
        split_on = ext.get("split_on")
        strip = ext.get("strip", True)
        tmp = df[[event_id_col, col]].rename(
            columns={event_id_col: "from_event_id", col: "to_disorder_type_name"}
        )
        tmp["to_disorder_type_name"] = tmp["to_disorder_type_name"].apply(
            lambda v: _maybe_split(v, split_on, strip)
        )
        tmp = tmp.explode("to_disorder_type_name", ignore_index=True)
        return tmp.dropna(subset=["to_disorder_type_name"]).reset_index(drop=True)

    def _rel_involved_in(
        self, spec: RelationshipSpec, df: pd.DataFrame, extractions: list[dict], event_id_col: str
    ) -> pd.DataFrame:
        parts = []
        for ext in extractions:
            col = ext["actor_column"]
            side = ext["side"]
            role = ext["role"]
            split_on = ext.get("split_on")
            strip = ext.get("strip", True)
            tmp = pd.DataFrame({
                "from_actor_name": df[col],
                "to_event_id": df[event_id_col],
            })
            tmp["from_actor_name"] = tmp["from_actor_name"].apply(
                lambda v: _maybe_split(v, split_on, strip)
            )
            tmp = tmp.explode("from_actor_name", ignore_index=True)
            tmp = tmp.dropna(subset=["from_actor_name"])
            tmp["side"] = side
            tmp["role"] = role
            parts.append(tmp)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return out.reset_index(drop=True)

    def _rel_has_type(self, spec: RelationshipSpec, df: pd.DataFrame, extraction: dict) -> pd.DataFrame:
        parts = []
        for pair in extraction["pairs"]:
            actor_col = pair["actor_column"]
            type_col = pair["type_column"]
            tmp = df[[actor_col, type_col]].dropna()
            tmp = tmp[(tmp[actor_col].astype(str).str.strip() != "") &
                      (tmp[type_col].astype(str).str.strip() != "")]
            tmp = tmp.rename(
                columns={actor_col: "from_actor_name", type_col: "to_actor_type_name"}
            )
            parts.append(tmp)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return out.drop_duplicates().reset_index(drop=True)
