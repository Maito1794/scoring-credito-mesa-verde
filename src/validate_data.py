"""Validación de calidad del dato que corre en el CI antes de entrenar.

Reglas duras (fallan el pipeline): esquema, dominio de categorías, objetivo binario,
volumen mínimo y porcentaje de nulos por columna.
Reglas blandas (solo avisan): duplicados y valores fuera de rango, que la limpieza trata.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.schema import CATEGORIES, PLAUSIBLE_RANGES, RAW_COLUMNS, TARGET

MIN_ROWS = 1_000
MAX_NULL_RATIO = 0.15  # loan_int_rate trae ~9.6 % de nulos en el dataset original


def validate(df: pd.DataFrame) -> dict:
    errors, warnings = [], []

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Faltan columnas: {missing}")
        return {"ok": False, "errors": errors, "warnings": warnings}

    if len(df) < MIN_ROWS:
        errors.append(f"Solo {len(df)} filas; el mínimo es {MIN_ROWS}")

    target_values = set(df[TARGET].dropna().unique())
    if not target_values <= {0, 1}:
        errors.append(f"{TARGET} tiene valores fuera de {{0, 1}}: {sorted(target_values)}")
    if df[TARGET].isna().any():
        errors.append(f"{TARGET} tiene nulos")

    for col, allowed in CATEGORIES.items():
        unknown = set(df[col].dropna().unique()) - set(allowed)
        if unknown:
            errors.append(f"{col} tiene categorías no esperadas: {sorted(unknown)}")

    for col in RAW_COLUMNS:
        ratio = float(df[col].isna().mean())
        if ratio > MAX_NULL_RATIO:
            errors.append(f"{col}: {ratio:.1%} de nulos (máximo {MAX_NULL_RATIO:.0%})")

    # La API calcula loan_percent_income = loan_amnt / person_income; si el dato no cuadra,
    # el modelo se entrenaría con un valor distinto del que recibe en producción
    derived = (df["loan_amnt"] / df["person_income"]).round(2)
    inconsistent = int(((derived - df["loan_percent_income"]).abs() > 0.011).sum())
    if inconsistent:
        warnings.append(
            f"loan_percent_income: {inconsistent} filas no cuadran con loan_amnt / person_income"
        )

    dups = int(df.duplicated().sum())
    if dups:
        warnings.append(f"{dups} filas duplicadas")
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        out = int((~df[col].between(low, high) & df[col].notna()).sum())
        if out:
            warnings.append(f"{col}: {out} valores fuera de [{low}, {high}]")

    return {"ok": not errors, "rows": int(len(df)), "errors": errors, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="data/credit_risk_dataset.csv")
    parser.add_argument("--report", default="outputs/data_validation.json")
    args = parser.parse_args()

    report = validate(pd.read_csv(args.csv))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
