# CriticalGraphRAG

Chatbot de análisis del conflicto Israel/Palestina basado en datos de [ACLED](https://acleddata.com/). Combina un Knowledge Graph en Neo4j con embeddings vectoriales y un agente LLM (Gemini) que decide en cada turno qué herramienta usar para responder.

## Arquitectura

```
Pregunta del usuario
        │
        ▼
   Agente LLM (Gemini 2.5 Pro)
        │
   ┌────┴──────────────────────────┐
   │            Decide             │
   ▼                ▼              ▼
cypher_query  similarity_search  text2cypher
(query        (búsqueda          (fallback:
predefinida)   vectorial +        LLM genera
               expansión          Cypher libre)
               de vecinos)
        │
        ▼
   Neo4j Knowledge Graph
```

### Tres tools del agente

| Tool | Qué hace | Cuándo se usa |
|------|----------|---------------|
| `cypher_query` | Ejecuta una query Cypher predefinida de `config/cypher_library.yaml` | Preguntas factuales con patrón conocido (conteos, rankings, agregaciones) |
| `similarity_search` | Búsqueda vectorial sobre `Event.notes` + expansión de vecinos en el grafo | Preguntas descriptivas, "qué ocurrió", búsqueda por contenido |
| `text2cypher` | LLM traduce la pregunta a Cypher usando el schema como contexto | Cuando ninguna query del library encaja |

### Knowledge Graph

Nodos: `Event`, `Month`, `EventType`, `DisorderType`, `Actor`, `ActorType`, `Location`, `Source`

Relaciones: `IN_MONTH`, `OF_TYPE`, `SUBTYPE_OF`, `OF_DISORDER`, `INVOLVED_IN`, `HAS_TYPE`, `AT_LOCATION`, `REPORTED_BY`

El schema completo está en `config/graph_schema.yaml` y es el **contrato del pipeline**: ningún label, relación ni propiedad se hardcodea en código.

### Embeddings

Default: **Gemini `gemini-embedding-001`** a 1536 dims (Matryoshka), cross-lingual (notas en EN, queries en ES/EN). Solo `Event.notes` se embebe; el resto del grafo se navega por relaciones.

---

## Requisitos

- Python 3.11+
- Docker (para Neo4j)
- Cuenta en [ACLED](https://acleddata.com/) (API key)
- Google API key (Gemini)

---

## Instalación

```bash
git clone <repo>
cd CriticalGraphRAG

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Editar .env con tus credenciales
```

---

## Variables de entorno (`.env`)

```env
# ACLED
ACLED_USERNAME=tu_email@ejemplo.com
ACLED_PASSWORD=tu_password

# Neo4j (levantado con docker-compose)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM y embeddings (default: Gemini)
GOOGLE_API_KEY=tu_google_api_key

# Opcional
LOG_LEVEL=INFO
```

---

## Ejecución de punta a punta

### 1. Levantar Neo4j

```bash
docker-compose up -d neo4j
```

### 2. Descargar dataset (una vez)

```bash
python scripts/01_download.py
```

Descarga un subset filtrado de ACLED (Israel/Palestina). Los filtros están en `config/dataset_filter.yaml`.

### 3. Construir el Knowledge Graph

```bash
# CSV → parquets de nodos y relaciones
python scripts/03_build_graph.py

# Calcular embeddings de Event.notes
python scripts/04_embeddings.py

# Cargar todo a Neo4j + crear vector index
python scripts/05_load_neo4j.py --mode destructive   # primera carga
python scripts/05_load_neo4j.py --mode incremental   # actualizaciones
```

Cada script persiste su salida en `data/graph/` para poder reiniciar desde cualquier checkpoint sin recomputar.

### 4. Levantar el chatbot

```bash
uvicorn web.app:app --reload
```

---

## Estructura del proyecto

```
CriticalGraphRAG/
├── config/
│   ├── dataset_filter.yaml       # filtros de descarga ACLED
│   ├── graph_schema.yaml         # contrato del KG (nodos, relaciones, atributos)
│   ├── embeddings.yaml           # proveedor activo + dimensiones
│   ├── agent.yaml                # modelos LLM + parámetros de retrieval
│   └── cypher_library.yaml       # queries predefinidas para cypher_query
├── data/
│   ├── acled_israel_2023.csv     # dataset (gitignored)
│   └── graph/
│       ├── nodes/{Label}.parquet
│       ├── relationships/{TYPE}.parquet
│       └── event_embeddings.parquet
├── src/
│   ├── acled_client.py           # cliente ACLED con OAuth2
│   ├── neo4j_conn.py             # driver Neo4j (basic / OAuth2)
│   ├── config.py                 # carga YAMLs + resolución de API keys
│   ├── schema.py                 # parser tipado de graph_schema.yaml
│   ├── embeddings/
│   │   ├── base.py               # interfaz EmbeddingProvider + factory
│   │   ├── google.py             # Gemini (default)
│   │   └── huggingface.py        # fallback local multilingüe
│   ├── graph/
│   │   ├── builder.py            # CSV + schema → DataFrames
│   │   └── neo4j_store.py        # interfaz GraphStore + implementación Neo4j
│   └── tools/
│       ├── similarity_search.py  # Neo4jVector + expansión de vecinos
│       ├── cypher_query.py       # CypherLibrary + StructuredTool
│       └── text2cypher.py        # GraphCypherQAChain con validación/retry
├── scripts/
│   ├── 01_download.py            # descarga vía API ACLED
│   ├── 02_explore.py             # EDA
│   ├── 03_build_graph.py         # CSV → parquets
│   ├── 04_embeddings.py          # parquets → embeddings.parquet
│   └── 05_load_neo4j.py          # carga Neo4j
├── web/                          # FastAPI + UI
├── tests/
│   └── cgrag_evaluation_questions.csv   # 50 preguntas de evaluación (ES)
├── requirements.txt
├── docker-compose.yaml
└── .env.example
```

---

## Cambiar de proveedor de embeddings

1. Editar `provider:` en `config/embeddings.yaml` (`google` | `huggingface`).
2. Agregar la API key correspondiente al `.env`.
3. Eliminar el índice vectorial en Neo4j (consola web).
4. Re-ejecutar `scripts/04_embeddings.py` y `scripts/05_load_neo4j.py`.

El proveedor HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dims) funciona sin API key y es útil como fallback local.

---

## Agregar queries predefinidas

Editar `config/cypher_library.yaml` sin tocar código. Cada entrada requiere:

```yaml
- id: "slug_unico"
  description: "Qué calcula en una línea"
  when_to_use: |
    Texto para el LLM: bajo qué preguntas conviene usar esta query.
  parameters:
    - name: "param"
      type: "string"
      description: "..."
      required: true
  returns: "columna: TIPO"
  cypher: |
    MATCH ...
    RETURN ...
```

---

## Tests

```bash
pytest tests/
```

El conjunto de evaluación (`tests/cgrag_evaluation_questions.csv`) contiene 50 preguntas en español y es la especificación funcional del sistema.

---

## Stack

| Componente | Tecnología |
|------------|------------|
| Base de datos de grafo | Neo4j 5+ |
| Embeddings | Gemini `gemini-embedding-001` (default) / HuggingFace (local) |
| Agente planificador | Gemini 2.5 Pro via LangGraph |
| Generación de Cypher | Gemini 2.5 Flash |
| Abstracción LLM/grafo | LangChain, LangGraph, langchain-neo4j |
| ETL | pandas + pyarrow (parquet) |
| Web | FastAPI + Jinja2 |
| Datos | ACLED (Armed Conflict Location & Event Data) |
