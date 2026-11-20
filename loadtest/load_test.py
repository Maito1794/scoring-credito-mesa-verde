"""Prueba de carga en flujo continuo contra /predict y comprobación del SLA.

Envía solicitudes a ritmo constante (lazo abierto: cada envío está programado en el tiempo y no
espera a la respuesta anterior), así una API lenta no reduce artificialmente la carga.
Mide latencia p50/p95/p99 y disponibilidad, y sale con código 1 si se incumple el SLA.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

# Permite importar src/ al ejecutar el script directamente desde la raíz del repo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.schema import PLAUSIBLE_RANGES  # noqa: E402

FIELDS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_amnt",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]


def load_payloads(csv: str, n: int, seed: int = 42) -> list[dict]:
    # Solicitudes reales del dataset, filtradas a valores que la API acepta
    df = pd.read_csv(csv)[FIELDS]
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col in df.columns:
            df = df[df[col].isna() | df[col].between(low, high)]
    df = df.sample(n=min(n, len(df)), random_state=seed, replace=len(df) < n)
    df = df.astype(object).where(df.notna(), None)
    return df.to_dict(orient="records")


async def one_call(client, url, payload, start_at, results):
    await asyncio.sleep(max(0.0, start_at - time.perf_counter()))
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload)
        ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
    results.append((ok, (time.perf_counter() - t0) * 1000))


async def run(base_url: str, rps: float, duration: float, payloads: list[dict], timeout: float):
    total = int(rps * duration)
    results: list[tuple[bool, float]] = []
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        t_start = time.perf_counter() + 0.5
        tasks = [
            one_call(
                client,
                f"{base_url}/predict",
                payloads[i % len(payloads)],
                t_start + i / rps,
                results,
            )
            for i in range(total)
        ]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t_start
    return results, elapsed


def summarize(results, elapsed, rps, p95_max, avail_min) -> dict:
    lat = np.array([ms for _, ms in results])
    ok = sum(1 for good, _ in results if good)
    summary = {
        "requests": len(results),
        "target_rps": rps,
        "achieved_rps": round(len(results) / elapsed, 1),
        "availability": round(ok / len(results), 5),
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "p99_ms": round(float(np.percentile(lat, 99)), 1),
        "sla": {"p95_ms_max": p95_max, "availability_min": avail_min},
    }
    summary["sla_ok"] = summary["p95_ms"] <= p95_max and summary["availability"] >= avail_min
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--csv", default="data/credit_risk_dataset.csv")
    parser.add_argument("--rps", type=float, default=50)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--p95-ms", type=float, default=200)
    parser.add_argument("--availability", type=float, default=0.999)
    parser.add_argument("--out", default="outputs/load_test.json")
    args = parser.parse_args()

    payloads = load_payloads(args.csv, n=2_000)
    results, elapsed = asyncio.run(run(args.url, args.rps, args.duration, payloads, args.timeout))
    summary = summarize(results, elapsed, args.rps, args.p95_ms, args.availability)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["sla_ok"] else 1)


if __name__ == "__main__":
    main()
