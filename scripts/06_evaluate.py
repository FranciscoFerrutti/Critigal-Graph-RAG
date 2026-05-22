"""
Script 06 — Evaluación del agente RAG sobre las preguntas de test.

Dos fases independientes (se pueden correr juntas o por separado):

  1. populate-gold:
     Ejecuta las queries Cypher de `tests/gold_cypher.yaml` contra Neo4j
     y deriva la respuesta canónica por entry según el `answer_format`.
     Sobrescribe la columna `respuesta_esperada` de `tests/gold_subset.csv`.

  2. run-eval:
     Invoca al `ChatbotAgent` por cada pregunta de `cgrag_evaluation_questions.csv`
     (50 preguntas) capturando respuesta, tool_calls y latencia.
     Para las 18 del subset gold compara la respuesta con la canónica con
     métricas (exact-match numérico ±5%, set-overlap para listas, substring sí/no).
     Para las 32 restantes solo guarda el output cualitativo.
     Escribe `tests/eval_results.csv`.

Uso:
    poetry run python scripts/06_evaluate.py             # ambas fases
    poetry run python scripts/06_evaluate.py --only-gold # solo populate
    poetry run python scripts/06_evaluate.py --only-eval # solo eval (gold previo)
    poetry run python scripts/06_evaluate.py --limit 5   # eval solo primeras 5 preguntas
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.neo4j_conn import get_driver

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
GOLD_YAML = ROOT / "tests" / "gold_cypher.yaml"
GOLD_CSV = ROOT / "tests" / "gold_subset.csv"
ALL_QUESTIONS_CSV = ROOT / "tests" / "cgrag_evaluation_questions.csv"
EVAL_OUT_CSV = ROOT / "tests" / "eval_results.csv"


# =============================================================================
# Fase 1 — Poblar ground truth
# =============================================================================

def _strip_comment(fmt: str) -> str:
    """Quita comentarios después de ' — ' (em-dash) en el answer_format."""
    if " — " in fmt:
        return fmt.split(" — ")[0].strip()
    return fmt.strip()


def derive_answer(rows: list[dict[str, Any]], fmt: str) -> str:
    """
    Reduce un result-set Neo4j a la respuesta canónica según `answer_format`.

    Formatos soportados:
      - scalar(col)              → valor único de la primera fila
      - row(col1, col2)          → "col1=v1, col2=v2" (primera fila)
      - list(col)                → "v1; v2; v3"
      - list((col1, col2))       → "col1=v1, col2=v2; col1=v3, col2=v4"
      - dict{key_col: val_col}   → "key1=val1; key2=val2"
      - event_count[A] - event_count[B]  → cálculo aritmético sobre la col 'month'
    """
    fmt = _strip_comment(fmt)
    if not rows:
        return "0" if fmt.startswith("scalar(") else ""

    # scalar
    m = re.fullmatch(r"scalar\(([^)]+)\)", fmt)
    if m:
        col = m.group(1).strip()
        v = rows[0].get(col)
        return "" if v is None else str(v)

    # row
    m = re.fullmatch(r"row\(([^)]+)\)", fmt)
    if m:
        cols = [c.strip() for c in m.group(1).split(",")]
        return ", ".join(f"{c}={rows[0].get(c)}" for c in cols)

    # list((col1, col2)) — composite list
    m = re.fullmatch(r"list\(\(([^)]+)\)\)", fmt)
    if m:
        cols = [c.strip() for c in m.group(1).split(",")]
        return "; ".join(", ".join(f"{c}={r.get(c)}" for c in cols) for r in rows)

    # list(col) — simple list
    m = re.fullmatch(r"list\(([^)]+)\)", fmt)
    if m:
        col = m.group(1).strip()
        return "; ".join(str(r.get(col)) for r in rows)

    # dict{k: v}
    m = re.fullmatch(r"dict\{([^:]+):\s*([^}]+)\}", fmt)
    if m:
        k_col, v_col = m.group(1).strip(), m.group(2).strip()
        return "; ".join(f"{r.get(k_col)}={r.get(v_col)}" for r in rows)

    # diff: event_count[A] - event_count[B]
    m = re.fullmatch(r"event_count\[([^\]]+)\]\s*-\s*event_count\[([^\]]+)\]", fmt)
    if m:
        a_key, b_key = m.group(1), m.group(2)
        by_m = {r.get("month") or r.get("month_id"): r.get("event_count") for r in rows}
        diff = (by_m.get(a_key, 0) or 0) - (by_m.get(b_key, 0) or 0)
        return str(diff)

    # fallback: dump bruto
    return json.dumps(rows, ensure_ascii=False, default=str)


def run_gold_query(session, cypher: str) -> list[dict[str, Any]]:
    result = session.run(cypher)
    out: list[dict[str, Any]] = []
    for record in result:
        row = {}
        for k, v in record.items():
            tname = type(v).__name__
            row[k] = str(v) if tname in {"Date", "DateTime"} else v
        out.append(row)
    return out


def populate_gold_truth() -> pd.DataFrame:
    """
    Lee tests/gold_cypher.yaml, ejecuta cada query y deriva la respuesta canónica.
    Sobrescribe la columna `respuesta_esperada` de tests/gold_subset.csv.

    Returns:
        DataFrame con el gold_subset actualizado.
    """
    with open(GOLD_YAML, "r", encoding="utf-8") as f:
        gold_yaml = yaml.safe_load(f)

    queries_by_id = {q["id"]: q for q in gold_yaml["queries"]}
    df = pd.read_csv(GOLD_CSV, dtype=str, keep_default_na=False)

    with get_driver() as driver, driver.session() as session:
        for idx, row in df.iterrows():
            qid = row["gold_query_id"]
            if qid not in queries_by_id:
                logger.warning("gold_query_id '%s' no está en gold_cypher.yaml — skip", qid)
                continue
            entry = queries_by_id[qid]
            rows = run_gold_query(session, entry["cypher"])
            answer = derive_answer(rows, entry["answer_format"])
            df.at[idx, "respuesta_esperada"] = answer
            logger.info("Gold %s: %s", qid, answer[:100])

    df.to_csv(GOLD_CSV, index=False)
    logger.info("✓ Ground truth escrito en %s", GOLD_CSV)
    return df


# =============================================================================
# Fase 2 — Evaluación del agente
# =============================================================================

def _normalize_number(text: str) -> float | None:
    """Extrae el primer número (entero o decimal) de un texto. None si no encuentra."""
    m = re.search(r"-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|-?\d+", text or "")
    if not m:
        return None
    raw = m.group(0)
    # heurística: si tiene tanto . como , asumir , como decimal style ES
    if "." in raw and "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"[^\w]+", (text or "").lower()) if t}


def score_answer(pred: str, expected: str, tipo: str) -> dict[str, Any]:
    """
    Calcula score (0..1) comparando la respuesta del agente con la canónica.
    Tres métodos:
      - numeric_match: para preguntas con respuesta numérica (±5% tolerancia)
      - set_overlap:   Jaccard sobre tokens (para listas, rankings)
      - substring_yn:  para sí/no (negativa/ausencia)
    """
    pred = (pred or "").strip()
    expected = (expected or "").strip()

    # Negativa/ausencia: testa Sí/No
    if "Negativa" in tipo or "Ausencia" in tipo:
        pl = pred.lower()
        # esperado: ninguno → respuesta debería decir "no"
        if not expected or expected.lower() in {"", "[]", "0"}:
            match = ("no" in pl) or ("ninguno" in pl) or ("ningún" in pl)
            return {"method": "substring_yn", "score": 1.0 if match else 0.0}
        # esperado: lista no vacía → respuesta debería decir "sí" o nombrar alguno
        tokens_exp = _tokenize(expected)
        tokens_pred = _tokenize(pred)
        overlap = len(tokens_exp & tokens_pred)
        score = overlap / max(len(tokens_exp), 1)
        return {"method": "substring_yn", "score": min(score * 2, 1.0)}

    # Numérico: si expected es número, comparar
    exp_n = _normalize_number(expected)
    pred_n = _normalize_number(pred)
    if exp_n is not None and pred_n is not None:
        if exp_n == 0:
            score = 1.0 if pred_n == 0 else 0.0
        else:
            err = abs(pred_n - exp_n) / abs(exp_n)
            score = 1.0 if err <= 0.05 else max(0.0, 1.0 - err)
        return {"method": "numeric_match", "score": float(score),
                "expected_num": exp_n, "pred_num": pred_n}

    # Listas y composites: Jaccard sobre tokens (ignora keys k= y v=)
    exp_tokens = _tokenize(expected)
    pred_tokens = _tokenize(pred)
    if not exp_tokens:
        return {"method": "set_overlap", "score": 0.0}
    inter = exp_tokens & pred_tokens
    union = exp_tokens | pred_tokens
    score = len(inter) / max(len(union), 1)
    return {"method": "set_overlap", "score": float(score),
            "intersection_size": len(inter), "union_size": len(union)}


def run_eval(limit: int | None = None) -> pd.DataFrame:
    """
    Corre el agente sobre todas las preguntas y guarda los resultados.
    Para las 18 gold compara respuesta vs ground truth.
    """
    # Cargar gold subset (con ground truth ya poblado)
    gold_df = pd.read_csv(GOLD_CSV)
    gold_by_text = {row["pregunta"]: row for _, row in gold_df.iterrows()}

    # También indexamos por id si está en el CSV original (no aplica aquí)
    all_df = pd.read_csv(ALL_QUESTIONS_CSV)

    # Combinamos: las preguntas a evaluar son las 50 del CSV original, pero
    # las 18 del subset gold pueden no estar (fueron reformuladas para Israel-2023).
    # Estrategia: evaluamos las 18 reformuladas + las 50 originales como cualitativo.
    rows_to_eval: list[dict[str, Any]] = []
    for _, row in gold_df.iterrows():
        rows_to_eval.append({
            "id": row["id"],
            "tipo": row["tipo"],
            "pregunta": row["pregunta"],
            "respuesta_esperada": row["respuesta_esperada"],
            "is_gold": True,
        })
    for _, row in all_df.iterrows():
        rows_to_eval.append({
            "id": f"orig_{row['id']}",
            "tipo": row["tipo"],
            "pregunta": row["pregunta"],
            "respuesta_esperada": row.get("respuesta_esperada", ""),
            "is_gold": False,
        })

    if limit:
        rows_to_eval = rows_to_eval[:limit]

    # Lazy-import del agente para que populate-gold funcione sin Gemini quota
    from src.agent import ChatbotAgent
    agent = ChatbotAgent()

    results: list[dict[str, Any]] = []
    for i, item in enumerate(rows_to_eval, 1):
        logger.info("[%d/%d] %s", i, len(rows_to_eval), item["pregunta"][:80])
        t0 = time.monotonic()
        try:
            trace = agent.invoke_trace(item["pregunta"])
            response = trace["response"]
            tool_calls = trace["tool_calls"]
            agent_error = ""
        except Exception as e:
            logger.exception("Falló invoke_trace")
            response = ""
            tool_calls = []
            agent_error = str(e)
        latency_ms = (time.monotonic() - t0) * 1000

        # Métrica solo para gold
        score_info = (
            score_answer(response, item["respuesta_esperada"], item["tipo"])
            if item["is_gold"]
            else {"method": "qualitative_only", "score": None}
        )

        results.append({
            "id": item["id"],
            "tipo": item["tipo"],
            "pregunta": item["pregunta"],
            "respuesta_esperada": item["respuesta_esperada"],
            "respuesta_modelo": response,
            "is_gold": item["is_gold"],
            "tool_calls": json.dumps(tool_calls, ensure_ascii=False, default=str),
            "n_tool_calls": len(tool_calls),
            "first_tool": tool_calls[0]["name"] if tool_calls else "",
            "score": score_info.get("score"),
            "score_method": score_info.get("method"),
            "latency_ms": round(latency_ms, 1),
            "agent_error": agent_error,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(EVAL_OUT_CSV, index=False)
    logger.info("✓ Resultados escritos en %s (%d filas)", EVAL_OUT_CSV, len(out_df))
    return out_df


def print_summary(df: pd.DataFrame) -> None:
    gold = df[df["is_gold"]]
    if gold.empty:
        print("\n(No hay filas gold para resumir.)")
    else:
        print("\n" + "=" * 70)
        print("RESUMEN MÉTRICAS — subset gold (18 preguntas)")
        print("=" * 70)
        print(f"  Accuracy media     : {gold['score'].mean():.3f}")
        print(f"  Accuracy mediana   : {gold['score'].median():.3f}")
        print(f"  Latencia media (ms): {gold['latency_ms'].mean():.0f}")
        print("\n  Por tipo de pregunta:")
        for tipo, sub in gold.groupby("tipo"):
            print(f"    {tipo:35s} n={len(sub):2d}  acc={sub['score'].mean():.3f}")
        print("\n  Tool más usada:")
        print("   ", gold["first_tool"].value_counts().to_dict())

    full = df
    print("\n" + "=" * 70)
    print("Distribución de tools sobre TODAS las preguntas evaluadas")
    print("=" * 70)
    print("  ", full["first_tool"].value_counts().to_dict())
    errs = full[full["agent_error"] != ""]
    if not errs.empty:
        print(f"\n  Errores del agente: {len(errs)} / {len(full)}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--only-gold", action="store_true",
                   help="Solo poblar ground truth; no correr el agente.")
    p.add_argument("--only-eval", action="store_true",
                   help="Solo correr el agente; asume ground truth ya poblado.")
    p.add_argument("--limit", type=int, default=None,
                   help="Evalúa solo las primeras N preguntas (debug).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.only_eval:
        logger.info("=== Fase 1: poblar ground truth ===")
        populate_gold_truth()

    if not args.only_gold:
        logger.info("=== Fase 2: evaluación del agente ===")
        results = run_eval(limit=args.limit)
        print_summary(results)


if __name__ == "__main__":
    main()
