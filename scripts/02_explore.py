"""
Script 02 — EDA (Exploratory Data Analysis) del dataset ACLED.

Carga el CSV descargado y genera un reporte en consola:
  - Dimensiones y columnas
  - Tipos de dato y valores nulos
  - Cardinalidades de columnas categóricas
  - Distribución de event_type y sub_event_type
  - Top actores (actor1 y actor2)
  - Rango de fechas y distribución anual
  - Estadísticas de fatalities
  - Distribución geográfica (admin1)

Uso:
    python scripts/02_explore.py
    python scripts/02_explore.py --input data/acled_colombia_2023_2024.csv
    python scripts/02_explore.py --sample   # usa datos de ejemplo hardcodeados
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Datos de ejemplo (para testear sin API key)
# ---------------------------------------------------------------------------

SAMPLE_DATA = [
    {
        "event_id_cnty": "COL12345",
        "event_date": "2023-03-15",
        "year": 2023,
        "time_precision": 1,
        "event_type": "Battles",
        "sub_event_type": "Armed clash",
        "actor1": "FARC Dissident Forces",
        "assoc_actor_1": "",
        "inter1": "2",
        "actor2": "Military Forces of Colombia (2010-)",
        "assoc_actor_2": "",
        "inter2": "1",
        "interaction": "12",
        "civilian_targeting": "",
        "iso": 170,
        "region": "South America",
        "country": "Colombia",
        "admin1": "Caquetá",
        "admin2": "Cartagena del Chairá",
        "admin3": "",
        "location": "Cartagena del Chairá",
        "latitude": 1.3317,
        "longitude": -74.8558,
        "geo_precision": 1,
        "source": "El Tiempo; Semana",
        "source_scale": "National",
        "notes": "Armed clash between FARC dissidents and military in rural Caquetá. Forces engaged near the Caguán river. No civilian casualties reported.",
        "fatalities": 3,
        "tags": "",
        "timestamp": 1678924800,
    },
    {
        "event_id_cnty": "COL12346",
        "event_date": "2023-05-22",
        "year": 2023,
        "time_precision": 1,
        "event_type": "Violence against civilians",
        "sub_event_type": "Attack",
        "actor1": "ELN: National Liberation Army",
        "assoc_actor_1": "",
        "inter1": "2",
        "actor2": "Civilians",
        "assoc_actor_2": "",
        "inter2": "7",
        "interaction": "27",
        "civilian_targeting": "Civilian targeting",
        "iso": 170,
        "region": "South America",
        "country": "Colombia",
        "admin1": "Norte de Santander",
        "admin2": "Tibú",
        "admin3": "",
        "location": "Tibú",
        "latitude": 8.6552,
        "longitude": -72.7325,
        "geo_precision": 1,
        "source": "El Colombiano",
        "source_scale": "National",
        "notes": "ELN members attacked a civilian community in Tibú, Norte de Santander, killing two farmers accused of working with rival groups.",
        "fatalities": 2,
        "tags": "",
        "timestamp": 1684713600,
    },
    {
        "event_id_cnty": "COL12347",
        "event_date": "2023-07-04",
        "year": 2023,
        "time_precision": 1,
        "event_type": "Protests",
        "sub_event_type": "Peaceful protest",
        "actor1": "Protesters (Colombia)",
        "assoc_actor_1": "Trade Union",
        "inter1": "5",
        "actor2": "",
        "assoc_actor_2": "",
        "inter2": "0",
        "interaction": "50",
        "civilian_targeting": "",
        "iso": 170,
        "region": "South America",
        "country": "Colombia",
        "admin1": "Bogotá",
        "admin2": "Bogotá",
        "admin3": "",
        "location": "Bogotá",
        "latitude": 4.7110,
        "longitude": -74.0721,
        "geo_precision": 1,
        "source": "El Espectador",
        "source_scale": "National",
        "notes": "Labor unions and social organizations marched in Bogotá demanding changes to government economic policy and protesting rising unemployment.",
        "fatalities": 0,
        "tags": "",
        "timestamp": 1688428800,
    },
    {
        "event_id_cnty": "COL12348",
        "event_date": "2023-09-10",
        "year": 2023,
        "time_precision": 1,
        "event_type": "Battles",
        "sub_event_type": "Government regains territory",
        "actor1": "Military Forces of Colombia (2010-)",
        "assoc_actor_1": "Police Forces of Colombia",
        "inter1": "1",
        "actor2": "Los Rastrojos",
        "assoc_actor_2": "",
        "inter2": "2",
        "interaction": "12",
        "civilian_targeting": "",
        "iso": 170,
        "region": "South America",
        "country": "Colombia",
        "admin1": "Nariño",
        "admin2": "Tumaco",
        "admin3": "",
        "location": "Tumaco",
        "latitude": 1.8073,
        "longitude": -78.8083,
        "geo_precision": 1,
        "source": "Semana; W Radio",
        "source_scale": "National",
        "notes": "Colombian security forces conducted an operation in Tumaco against Los Rastrojos criminal organization, retaking control of two neighborhoods.",
        "fatalities": 5,
        "tags": "",
        "timestamp": 1694304000,
    },
    {
        "event_id_cnty": "COL12349",
        "event_date": "2023-11-28",
        "year": 2023,
        "time_precision": 1,
        "event_type": "Explosions/Remote violence",
        "sub_event_type": "IED/Improvised explosive device",
        "actor1": "FARC Dissident Forces",
        "assoc_actor_1": "",
        "inter1": "2",
        "actor2": "Military Forces of Colombia (2010-)",
        "assoc_actor_2": "",
        "inter2": "1",
        "interaction": "12",
        "civilian_targeting": "",
        "iso": 170,
        "region": "South America",
        "country": "Colombia",
        "admin1": "Putumayo",
        "admin2": "Puerto Asís",
        "admin3": "",
        "location": "Puerto Asís",
        "latitude": 0.5028,
        "longitude": -76.5000,
        "geo_precision": 1,
        "source": "RCN Radio",
        "source_scale": "National",
        "notes": "IED explosion targeted a military patrol in Puerto Asís, Putumayo. Four soldiers wounded, no fatalities.",
        "fatalities": 0,
        "tags": "",
        "timestamp": 1701129600,
    },
    {
        "event_id_cnty": "COL12350",
        "event_date": "2024-01-15",
        "year": 2024,
        "time_precision": 1,
        "event_type": "Battles",
        "sub_event_type": "Armed clash",
        "actor1": "ELN: National Liberation Army",
        "assoc_actor_1": "",
        "inter1": "2",
        "actor2": "FARC Dissident Forces",
        "assoc_actor_2": "",
        "inter2": "2",
        "interaction": "22",
        "civilian_targeting": "",
        "iso": 170,
        "region": "South America",
        "country": "Colombia",
        "admin1": "Arauca",
        "admin2": "Arauquita",
        "admin3": "",
        "location": "Arauquita",
        "latitude": 7.0272,
        "longitude": -71.4330,
        "geo_precision": 1,
        "source": "Caracol Radio; El Tiempo",
        "source_scale": "National",
        "notes": "Clashes between ELN and FARC dissident forces in Arauquita, Arauca. Dispute over territorial control in the border region with Venezuela.",
        "fatalities": 7,
        "tags": "",
        "timestamp": 1705276800,
    },
    {
        "event_id_cnty": "COL12351",
        "event_date": "2024-02-20",
        "year": 2024,
        "time_precision": 1,
        "event_type": "Violence against civilians",
        "sub_event_type": "Abduction/forced disappearance",
        "actor1": "Gulf Clan",
        "assoc_actor_1": "",
        "inter1": "2",
        "actor2": "Civilians",
        "assoc_actor_2": "",
        "inter2": "7",
        "interaction": "27",
        "civilian_targeting": "Civilian targeting",
        "iso": 170,
        "region": "South America",
        "country": "Colombia",
        "admin1": "Chocó",
        "admin2": "Quibdó",
        "admin3": "",
        "location": "Quibdó",
        "latitude": 5.6919,
        "longitude": -76.6583,
        "geo_precision": 1,
        "source": "Verdad Abierta",
        "source_scale": "National",
        "notes": "Gulf Clan members abducted three community leaders in Quibdó, Chocó. The victims were later released after community negotiations.",
        "fatalities": 0,
        "tags": "",
        "timestamp": 1708387200,
    },
    {
        "event_id_cnty": "COL12352",
        "event_date": "2024-04-05",
        "year": 2024,
        "time_precision": 1,
        "event_type": "Protests",
        "sub_event_type": "Violent demonstration",
        "actor1": "Protesters (Colombia)",
        "assoc_actor_1": "Indigenous Group (Colombia)",
        "inter1": "5",
        "actor2": "Police Forces of Colombia",
        "assoc_actor_2": "",
        "inter2": "1",
        "interaction": "15",
        "civilian_targeting": "",
        "iso": 170,
        "region": "South America",
        "country": "Colombia",
        "admin1": "Cauca",
        "admin2": "Popayán",
        "admin3": "",
        "location": "Popayán",
        "latitude": 2.4419,
        "longitude": -76.6071,
        "geo_precision": 1,
        "source": "El País Cali; Semana",
        "source_scale": "National",
        "notes": "Indigenous groups and social organizations clashed with police in Popayán during a demonstration against extractive industries in indigenous territories.",
        "fatalities": 0,
        "tags": "",
        "timestamp": 1712275200,
    },
    {
        "event_id_cnty": "COL12353",
        "event_date": "2024-06-18",
        "year": 2024,
        "time_precision": 1,
        "event_type": "Battles",
        "sub_event_type": "Armed clash",
        "actor1": "Military Forces of Colombia (2010-)",
        "assoc_actor_1": "",
        "inter1": "1",
        "actor2": "ELN: National Liberation Army",
        "assoc_actor_2": "",
        "inter2": "2",
        "interaction": "12",
        "civilian_targeting": "",
        "iso": 170,
        "region": "South America",
        "country": "Colombia",
        "admin1": "Chocó",
        "admin2": "Bojayá",
        "admin3": "",
        "location": "Bojayá",
        "latitude": 6.9997,
        "longitude": -77.0003,
        "geo_precision": 2,
        "source": "El Tiempo; Defensoría del Pueblo",
        "source_scale": "National",
        "notes": "Armed clashes between Colombian military and ELN in the Bojayá area, Chocó. Displacement of civilian population reported in surrounding communities.",
        "fatalities": 2,
        "tags": "",
        "timestamp": 1718668800,
    },
    {
        "event_id_cnty": "COL12354",
        "event_date": "2024-08-30",
        "year": 2024,
        "time_precision": 1,
        "event_type": "Strategic developments",
        "sub_event_type": "Looting/property destruction",
        "actor1": "Gulf Clan",
        "assoc_actor_1": "",
        "inter1": "2",
        "actor2": "Civilians",
        "assoc_actor_2": "",
        "inter2": "7",
        "interaction": "27",
        "civilian_targeting": "Civilian targeting",
        "iso": 170,
        "region": "South America",
        "country": "Colombia",
        "admin1": "Antioquia",
        "admin2": "Turbo",
        "admin3": "",
        "location": "Turbo",
        "latitude": 8.0997,
        "longitude": -76.7264,
        "geo_precision": 1,
        "source": "Caracol Radio",
        "source_scale": "National",
        "notes": "Gulf Clan imposed a curfew and looted businesses in Turbo, Antioquia during a 48-hour armed strike (paro armado).",
        "fatalities": 0,
        "tags": "",
        "timestamp": 1725062400,
    },
]


# ---------------------------------------------------------------------------
# Funciones de análisis
# ---------------------------------------------------------------------------

SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def report_overview(df: pd.DataFrame) -> None:
    section("OVERVIEW")
    print(f"Filas:    {len(df):,}")
    print(f"Columnas: {len(df.columns)}")
    print(f"\nColumnas disponibles:\n  {list(df.columns)}")


def report_types_and_nulls(df: pd.DataFrame) -> None:
    section("TIPOS DE DATO Y VALORES NULOS")
    stats = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "nulls": df.isnull().sum(),
            "null_%": (df.isnull().sum() / len(df) * 100).round(1),
            "unique": df.nunique(),
        }
    )
    print(stats.to_string())


def report_date_range(df: pd.DataFrame) -> None:
    section("RANGO DE FECHAS")
    if "event_date" not in df.columns:
        print("  Columna 'event_date' no encontrada.")
        return

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    print(f"  Fecha mínima: {df['event_date'].min().date()}")
    print(f"  Fecha máxima: {df['event_date'].max().date()}")

    if "year" in df.columns:
        print("\n  Distribución por año:")
        print(df["year"].value_counts().sort_index().to_string())


def report_event_types(df: pd.DataFrame) -> None:
    section("DISTRIBUCIÓN DE event_type")
    if "event_type" not in df.columns:
        print("  Columna 'event_type' no encontrada.")
        return
    print(df["event_type"].value_counts().to_string())

    if "sub_event_type" in df.columns:
        print("\n  sub_event_type (top 15):")
        print(df["sub_event_type"].value_counts().head(15).to_string())


def report_top_actors(df: pd.DataFrame, top_n: int = 15) -> None:
    section(f"TOP {top_n} ACTORES")
    actors = []
    for col in ["actor1", "actor2"]:
        if col in df.columns:
            actors.append(df[col].dropna())

    if not actors:
        print("  Columnas 'actor1'/'actor2' no encontradas.")
        return

    combined = pd.concat(actors)
    combined = combined[combined.str.strip() != ""]
    top = combined.value_counts().head(top_n)
    print(top.to_string())


def report_fatalities(df: pd.DataFrame) -> None:
    section("ESTADÍSTICAS DE FATALITIES")
    if "fatalities" not in df.columns:
        print("  Columna 'fatalities' no encontrada.")
        return

    fat = pd.to_numeric(df["fatalities"], errors="coerce")
    print(f"  Total:    {fat.sum():,.0f}")
    print(f"  Media:    {fat.mean():.2f}")
    print(f"  Mediana:  {fat.median():.1f}")
    print(f"  Máximo:   {fat.max():.0f}")
    print(f"  Eventos con fatalities > 0: {(fat > 0).sum()} ({(fat > 0).mean()*100:.1f}%)")


def report_geography(df: pd.DataFrame, top_n: int = 15) -> None:
    section(f"DISTRIBUCIÓN GEOGRÁFICA (admin1, top {top_n})")
    if "admin1" not in df.columns:
        print("  Columna 'admin1' no encontrada.")
        return
    print(df["admin1"].value_counts().head(top_n).to_string())


def report_cardinalities(df: pd.DataFrame) -> None:
    section("CARDINALIDADES DE COLUMNAS CATEGÓRICAS")
    categorical_cols = [
        "event_type", "sub_event_type", "actor1", "actor2",
        "country", "admin1", "admin2", "region",
        "inter1", "inter2", "source_scale",
    ]
    existing = [c for c in categorical_cols if c in df.columns]
    card = {col: df[col].nunique() for col in existing}
    for col, n in card.items():
        print(f"  {col:<25} {n:>6} valores únicos")


def report_schema_suggestions(df: pd.DataFrame) -> None:
    section("SUGERENCIAS PARA graph_schema.yaml")
    lines = [
        "",
        "  Nodos candidatos detectados:",
        "    - Event    : event_id_cnty, event_date, event_type, sub_event_type,",
        "                 notes, fatalities, source, source_scale",
        "    - Actor    : actor1, actor2 (deduplicar por nombre)",
        "                 inter1, inter2 (tipo de actor)",
        "    - Location : location, admin1, admin2, admin3, country,",
        "                 latitude, longitude",
        "",
        "  Relaciones candidatas:",
        "    - (Actor)-[:INVOLVED_IN {role}]->(Event)",
        "        actor1 => role: 'perpetrator'",
        "        actor2 => role: 'second_party'",
        "    - (Event)-[:OCCURRED_AT]->(Location)",
        "",
        "  Campos para embeddings:",
        "    - Event    : event_type + sub_event_type + notes",
        "    - Actor    : actor_name",
        "    - Location : location_name + admin1 + country",
        "",
        "  Columna ID real de ACLED: 'event_id_cnty' (no 'data_id')",
        "    => Usar source_id_column: 'event_id_cnty' en graph_schema.yaml",
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA del dataset ACLED")
    parser.add_argument(
        "--input",
        default="data/acled_colombia_2023_2024.csv",
        help="Ruta al CSV de datos descargados",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Usar datos de ejemplo hardcodeados (no requiere CSV)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Cantidad de valores top a mostrar en rankings",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    if args.sample:
        logger.info("Usando datos de ejemplo hardcodeados (%d eventos)", len(SAMPLE_DATA))
        df = pd.DataFrame(SAMPLE_DATA)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(
                "Archivo no encontrado: %s\n"
                "  Ejecutá primero: python scripts/01_download.py\n"
                "  O usá --sample para datos de ejemplo.",
                input_path,
            )
            sys.exit(1)
        logger.info("Cargando: %s", input_path)
        df = pd.read_csv(input_path, low_memory=False)

    print(f"\n{'#'*70}")
    print(f"  REPORTE EDA — ACLED Colombia 2023-2024")
    print(f"{'#'*70}")

    report_overview(df)
    report_types_and_nulls(df)
    report_date_range(df)
    report_event_types(df)
    report_top_actors(df, top_n=args.top)
    report_fatalities(df)
    report_geography(df, top_n=args.top)
    report_cardinalities(df)
    report_schema_suggestions(df)

    print(f"\n{'#'*70}")
    print("  FIN DEL REPORTE")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
