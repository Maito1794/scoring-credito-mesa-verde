import json
import os

import numpy as np
import pandas as pd
import pytest

from src.schema import CATEGORIES, FEATURES, TARGET


def make_synthetic(n: int = 2_000, seed: int = 0) -> pd.DataFrame:
    """Datos ficticios con el mismo esquema que credit_risk_dataset.csv (solo para tests)."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.9, 0.5, n).round(0)
    amount = rng.integers(500, 35_000, n).astype(float)
    df = pd.DataFrame(
        {
            "person_age": rng.integers(20, 70, n),
            "person_income": income,
            "person_home_ownership": rng.choice(CATEGORIES["person_home_ownership"], n),
            "person_emp_length": rng.integers(0, 30, n).astype(float),
            "loan_intent": rng.choice(CATEGORIES["loan_intent"], n),
            "loan_grade": rng.choice(CATEGORIES["loan_grade"], n),
            "loan_amnt": amount,
            "loan_int_rate": rng.uniform(5, 23, n).round(2),
            "loan_percent_income": np.clip(amount / income, 0, 1).round(2),
            "cb_person_default_on_file": rng.choice(["Y", "N"], n, p=[0.2, 0.8]),
            "cb_person_cred_hist_length": rng.integers(2, 30, n),
        }
    )
    logit = -2.2 + 4 * df["loan_percent_income"] + (df["person_home_ownership"] == "RENT") * 0.8
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, synthetic_df):
    import joblib

    from src.features import add_dsti
    from src.train import build_pipeline

    out = tmp_path_factory.mktemp("model")
    x = add_dsti(synthetic_df)[FEATURES]
    model = build_pipeline("xgboost").fit(x, synthetic_df[TARGET])
    joblib.dump(model, out / "model.joblib")
    metrics = {"model_version": "test", "gini": 0.5}
    (out / "metrics.json").write_text(json.dumps(metrics))
    return out


@pytest.fixture()
def client(model_dir, tmp_path, monkeypatch):
    """Cliente de la API con un modelo de prueba; la política se puede editar en cada test."""
    from fastapi.testclient import TestClient

    import src.api as api

    policy = json.loads(open("config/policy.json", encoding="utf-8").read())
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    monkeypatch.setattr(api, "MODEL_DIR", model_dir)
    monkeypatch.setattr(api, "POLICY_PATH", policy_path)
    with TestClient(api.app) as c:
        c.policy_path = policy_path
        yield c


os.environ.setdefault("APP_VERSION", "test")
