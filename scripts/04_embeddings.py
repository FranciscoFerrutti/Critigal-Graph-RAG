"""
Script 04 — Cálculo de embeddings sobre Event.notes.

Lee:
  - data/graph/nodes/Event.parquet    (salida de 03_build_graph.py)
  - config/embeddings.yaml

Escribe:
  - data/graph/event_embeddings.parquet   (event_id, embedding)

Diseño:
  - Se saltea notes vacías (NaN o string vacío).
  - Batch size configurable desde embeddings.yaml.
  - Persiste como parquet para que 05_load_neo4j.py lo consuma sin recomputar.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_yaml
from src.embeddings import get_embedding_provider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calcula embeddings de Event.notes y guarda parquet.")
    p.add_argument("--nodes-dir", default="data/graph/nodes", help="Carpeta con los parquet de nodos.")
    p.add_argument("--embeddings-config", default="config/embeddings.yaml")
    p.add_argument("--out", default="data/graph/event_embeddings.parquet")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override del batch_size de embeddings.yaml (útil para rate limits).")
    p.add_argument("--rpm-limit", type=int, default=95,
                   help="Margen seguro bajo el RPM del proveedor (Gemini free: 100).")
    p.add_argument("--tpm-limit", type=int, default=28000,
                   help="Margen seguro bajo el TPM del proveedor (Gemini free: 30K).")
    p.add_argument("--rpd-limit", type=int, default=0,
                   help="Si > 0, abortar limpio al llegar a este nro de requests (Gemini free: 1000).")
    p.add_argument("--max-retries", type=int, default=6,
                   help="Reintentos por batch ante 429/5xx con backoff exponencial.")
    p.add_argument("--resume", action="store_true",
                   help="Si el parquet de salida existe, continuar desde donde quedó.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


class RateLimiter:
    """
    Rate limiter con ventana deslizante de 60s para RPM + TPM.

    Antes de cada request, espera hasta que el budget de RPM y TPM
    permitan el batch entrante. Más eficiente que un sleep fijo:
    aprovecha al máximo la cuota sin tirar 429.
    """

    def __init__(self, rpm: int, tpm: int, log) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self.log = log
        # cada elemento: (timestamp, tokens_de_la_request)
        self.window: deque[tuple[float, int]] = deque()

    def _purge(self, now: float) -> None:
        while self.window and now - self.window[0][0] >= 60.0:
            self.window.popleft()

    def acquire(self, batch_tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._purge(now)
            req_count = len(self.window)
            tok_sum = sum(t for _, t in self.window)
            if req_count < self.rpm and tok_sum + batch_tokens <= self.tpm:
                self.window.append((now, batch_tokens))
                return
            # esperar hasta que expire el evento más antiguo
            wait = 60.0 - (now - self.window[0][0]) + 0.1
            self.log.info(
                "Rate limit: req=%d/%d tok=%d/%d, esperando %.1fs.",
                req_count, self.rpm, tok_sum, self.tpm, wait,
            )
            time.sleep(max(wait, 0.5))


def estimate_tokens(texts: list[str]) -> int:
    """Estimación rápida: chars/4. Suficiente para batches del orden de 100 textos."""
    return sum(len(t) for t in texts) // 4


def embed_with_retry(embedder, texts: list[str], max_retries: int, log) -> list[list[float]]:
    """
    Llama a embed_documents con backoff exponencial ante rate limit o errores transitorios.
    Los SDK de Google levantan distintos tipos de excepción según versión; detectamos
    por texto del error.
    """
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            return embedder.embed_documents(texts)
        except Exception as e:
            msg = str(e).lower()
            is_retryable = (
                "429" in msg or "rate" in msg or "quota" in msg
                or "resource_exhausted" in msg or "503" in msg or "500" in msg
                or "deadline" in msg or "unavailable" in msg
            )
            if not is_retryable or attempt == max_retries:
                raise
            log.warning(
                "Batch falló (intento %d/%d): %s. Esperando %.1fs y reintentando.",
                attempt, max_retries, type(e).__name__, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    # nunca llega acá
    return []


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

    log.info("=== Embeddings (04) — inicio ===")
    cfg = load_yaml(args.embeddings_config)
    provider = get_embedding_provider(cfg)
    log.info(
        "Provider=%s model=%s dims=%d batch=%d",
        provider.provider_name, provider.model, provider.dimensions, provider.batch_size,
    )

    events_path = Path(args.nodes_dir) / "Event.parquet"
    df = pd.read_parquet(events_path)
    log.info("Events cargados: %d", len(df))

    mask = df["notes"].notna() & (df["notes"].astype(str).str.strip() != "")
    df = df.loc[mask, ["event_id", "notes"]].reset_index(drop=True)
    log.info("Events con notes no vacías: %d", len(df))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --resume: leer los event_ids ya procesados y saltarlos
    done_ids: set[str] = set()
    existing_rows: list[dict] = []
    if args.resume and out_path.exists():
        prev = pd.read_parquet(out_path)
        done_ids = set(prev["event_id"].tolist())
        existing_rows = prev.to_dict(orient="records")
        log.info("Resume: %d embeddings ya calculados, se saltan.", len(done_ids))
        df = df[~df["event_id"].isin(done_ids)].reset_index(drop=True)
        log.info("Events pendientes: %d", len(df))

    embedder = provider.get_document_embeddings()
    batch_size = args.batch_size or provider.batch_size
    log.info("Batch=%d rpm=%d tpm=%d max_retries=%d",
             batch_size, args.rpm_limit, args.tpm_limit, args.max_retries)
    limiter = RateLimiter(rpm=args.rpm_limit, tpm=args.tpm_limit, log=log)

    all_rows: list[dict] = list(existing_rows)
    processed_in_run = 0
    requests_in_run = 0
    try:
        for start in range(0, len(df), batch_size):
            if args.rpd_limit > 0 and requests_in_run >= args.rpd_limit:
                log.warning(
                    "RPD limit alcanzado (%d). Abortando limpio. "
                    "Re-ejecutar con --resume mañana.", args.rpd_limit,
                )
                break
            chunk = df.iloc[start : start + batch_size]
            texts = chunk["notes"].astype(str).tolist()
            limiter.acquire(estimate_tokens(texts))
            vectors = embed_with_retry(
                embedder, texts,
                max_retries=args.max_retries, log=log,
            )
            requests_in_run += 1
            for eid, vec in zip(chunk["event_id"].tolist(), vectors):
                all_rows.append({"event_id": eid, "embedding": vec})
            processed_in_run += len(chunk)
            log.info("Embeddings: %d / %d (req=%d)",
                     processed_in_run, len(df), requests_in_run)
    except Exception:
        # checkpoint defensivo: guardar lo que haya antes de re-lanzar
        log.exception("Falló el cálculo de embeddings. Guardando checkpoint parcial.")
        if all_rows:
            pd.DataFrame(all_rows).to_parquet(out_path, index=False)
            log.info("Checkpoint parcial guardado en %s (%d filas). "
                     "Re-ejecutá con --resume para continuar.", out_path, len(all_rows))
        raise

    out_df = pd.DataFrame(all_rows)
    out_df.to_parquet(out_path, index=False)
    log.info("Guardado: %s (%d filas)", out_path, len(out_df))
    log.info("=== Embeddings (04) — fin ===")


if __name__ == "__main__":
    main()
