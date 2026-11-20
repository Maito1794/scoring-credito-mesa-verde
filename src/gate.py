"""Puerta de calidad del modelo: el pipeline falla si el modelo nuevo no alcanza el mínimo.

1. Gini >= GINI_MIN (umbral absoluto que fija el Data Scientist).
2. Si existe un campeón en producción, el retador no puede perder más de MAX_DROP puntos de Gini.
3. Un modelo complejo solo se justifica si supera a la regresión logística de referencia (baseline).
"""

import argparse
import json
import os
import sys
from pathlib import Path


def evaluate(metrics: dict, gini_min: float, champion: dict | None, max_drop: float) -> list[str]:
    failures = []
    gini = float(metrics["gini"])
    if gini < gini_min:
        failures.append(f"Gini {gini:.4f} < mínimo {gini_min:.4f}")
    baseline = metrics.get("baseline")
    if baseline and metrics.get("algorithm") != baseline["algorithm"]:
        if gini <= float(baseline["gini"]):
            failures.append(
                f"Gini {gini:.4f} no supera a la referencia {baseline['algorithm']} "
                f"({float(baseline['gini']):.4f})"
            )
    if champion is not None:
        champ = float(champion["gini"])
        if gini < champ - max_drop:
            failures.append(
                f"Gini {gini:.4f} cae más de {max_drop:.2f} frente al campeón ({champ:.4f})"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="model/metrics.json")
    parser.add_argument("--champion", default="config/champion_metrics.json")
    parser.add_argument("--gini-min", type=float, default=float(os.getenv("GINI_MIN", "0.70")))
    parser.add_argument("--max-drop", type=float, default=float(os.getenv("GINI_MAX_DROP", "0.03")))
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    champ_path = Path(args.champion)
    champion = json.loads(champ_path.read_text(encoding="utf-8")) if champ_path.exists() else None

    failures = evaluate(metrics, args.gini_min, champion, args.max_drop)
    status = "RECHAZADO" if failures else "APROBADO"
    lines = [
        f"### Puerta de modelo: {status}",
        f"- Gini: {metrics['gini']} (mínimo {args.gini_min})",
        f"- AUC: {metrics['auc']} · KS: {metrics['ks']}",
        f"- Referencia: {metrics['baseline']['gini'] if metrics.get('baseline') else 'no hay'}",
        f"- Campeón: {champion['gini'] if champion else 'no hay'}",
        *[f"- ❌ {f}" for f in failures],
    ]
    print("\n".join(lines))
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
