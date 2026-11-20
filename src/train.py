"""Entrenamiento del modelo de scoring (basado en Credit_Score_Model.ipynb del Data Scientist).

Contrato con la API y el CI:
  - model/model.joblib: Pipeline de scikit-learn que recibe un DataFrame con FEATURES
  - model/metrics.json: model_version, auc, gini, ks, brier, baseline y features
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features import add_dsti
from src.schema import (
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    DSTI_ANNUAL_RATE,
    DSTI_TERM_MONTHS,
    FEATURES,
    NUMERIC_FEATURES,
    PLAUSIBLE_RANGES,
    TARGET,
)

SEED = 2
PROD_SIZE = 0.2  # se reserva para simular producción y deriva; el modelo nunca lo ve
TEST_SIZE = 0.2  # del 80 % restante, para medir el modelo
XGB_PARAMS = {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05}
METRICS_THRESHOLD = 0.25  # PD a partir de la cual se cuenta como impago para precision y recall
INPUT_FEATURES = [f for f in FEATURES if f not in DERIVED_FEATURES]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Quita duplicados y valores fuera de PLAUSIBLE_RANGES (los nulos se imputan)."""
    df = df.drop_duplicates()
    # Mismo cálculo que hace la API, para que entrenamiento y producción vean el mismo valor
    df = df.assign(loan_percent_income=(df["loan_amnt"] / df["person_income"]).round(2))
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        df = df[df[col].isna() | df[col].between(low, high)]
    return df.reset_index(drop=True)


def build_pipeline(algorithm: str = "xgboost") -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("enc", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    if algorithm == "xgboost":
        # n_jobs=1: en la API se predice de una en una y los hilos extra solo suman latencia
        clf = XGBClassifier(**XGB_PARAMS, random_state=SEED, eval_metric="logloss", n_jobs=1)
    elif algorithm == "logistic":
        clf = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    scores = model.predict_proba(x)[:, 1]
    auc = float(roc_auc_score(y, scores))
    fpr, tpr, _ = roc_curve(y, scores)
    preds = (scores >= METRICS_THRESHOLD).astype(int)
    return {
        "auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(float(np.max(tpr - fpr)), 4),
        "brier": round(float(brier_score_loss(y, scores)), 4),
        "precision": round(float(precision_score(y, preds)), 4),
        "recall": round(float(recall_score(y, preds)), 4),
    }


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/credit_risk_dataset.csv")
    parser.add_argument("--out", default="model")
    parser.add_argument("--version", default="dev")
    parser.add_argument("--algorithm", default="xgboost", choices=["xgboost", "logistic"])
    args = parser.parse_args()

    df = basic_clean(pd.read_csv(args.data))
    x, y = df[INPUT_FEATURES], df[TARGET]
    x_hist, _, y_hist, _ = train_test_split(
        x, y, test_size=PROD_SIZE, random_state=SEED, stratify=y
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x_hist, y_hist, test_size=TEST_SIZE, random_state=SEED, stratify=y_hist
    )
    x_train = add_dsti(x_train)[FEATURES]
    x_test = add_dsti(x_test)[FEATURES]

    model = build_pipeline(args.algorithm).fit(x_train, y_train)
    baseline = build_pipeline("logistic").fit(x_train, y_train)

    # La versión del modelo se identifica por la versión del código y la fecha de ejecución
    metrics = {
        "model_version": args.version,
        "git_sha": git_sha(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": args.algorithm,
        "seed": SEED,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        **evaluate(model, x_test, y_test),
        "baseline": {"algorithm": "logistic", **evaluate(baseline, x_test, y_test)},
        "metrics_threshold": METRICS_THRESHOLD,
        "dsti_term_months": DSTI_TERM_MONTHS,
        "dsti_annual_rate": DSTI_ANNUAL_RATE,
        "features": FEATURES,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
