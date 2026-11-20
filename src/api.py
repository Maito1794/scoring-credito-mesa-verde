"""API de scoring: /predict devuelve PD, DSTI, rating y decisión; /health informa de la versión."""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import add_dsti
from src.schema import CATEGORIES, FEATURES, PLAUSIBLE_RANGES

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
POLICY_PATH = Path(os.getenv("POLICY_PATH", "config/policy.json"))
APP_VERSION = os.getenv("APP_VERSION", "dev")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("scoring")
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # El modelo y la política se cargan una sola vez al arrancar el contenedor
    state["model"] = joblib.load(MODEL_DIR / "model.joblib")
    state["metrics"] = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    state["policy"] = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    yield
    state.clear()


app = FastAPI(title="Scoring de crédito", version=APP_VERSION, lifespan=lifespan)


def _range(col: str) -> dict:
    low, high = PLAUSIBLE_RANGES[col]
    return {"ge": low, "le": high}


class Solicitud(BaseModel):
    # Rangos y categorías salen de schema.py: la API rechaza lo mismo que descarta el entrenamiento
    person_age: int = Field(**_range("person_age"))
    person_income: float = Field(**_range("person_income"))
    person_home_ownership: Literal[tuple(CATEGORIES["person_home_ownership"])]
    person_emp_length: float | None = Field(default=None, **_range("person_emp_length"))
    loan_intent: Literal[tuple(CATEGORIES["loan_intent"])]
    loan_amnt: float = Field(**_range("loan_amnt"))
    cb_person_default_on_file: Literal[tuple(CATEGORIES["cb_person_default_on_file"])]
    cb_person_cred_hist_length: int = Field(**_range("cb_person_cred_hist_length"))


class Respuesta(BaseModel):
    pd: float  # PD con margen de conservadurismo: es la que decide
    pd_modelo: float  # PD tal como sale del modelo
    dsti: float
    rating: str
    decision: Literal["APROBAR", "REVISAR", "RECHAZAR"]
    motivo: str
    model_version: str
    app_version: str


def rating_for(pd_value: float, bands: list[dict]) -> str:
    for band in sorted(bands, key=lambda b: b["max_pd"]):
        if pd_value <= band["max_pd"]:
            return band["rating"]
    return bands[-1]["rating"]


def decision_for(pd_value: float, dsti: float, intent: str, policy: dict) -> tuple[str, str]:
    # Un hotfix puede endurecer los cortes de una finalidad concreta sin reentrenar
    cuts = {**policy["cutoffs"], **policy.get("overrides", {}).get(intent, {})}
    if dsti > cuts["max_dsti"]:
        return "RECHAZAR", f"DSTI {dsti:.1%} supera el máximo de {cuts['max_dsti']:.0%}"
    if pd_value < cuts["approve_below"]:
        return "APROBAR", f"PD {pd_value:.1%} por debajo del corte de {cuts['approve_below']:.0%}"
    if pd_value >= cuts["reject_from"]:
        return "RECHAZAR", f"PD {pd_value:.1%} desde el corte de {cuts['reject_from']:.0%}"
    return "REVISAR", f"PD {pd_value:.1%} en la zona de revisión"


@app.get("/health")
def health() -> dict:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {
        "status": "ok",
        "model_version": state["metrics"]["model_version"],
        "app_version": APP_VERSION,
        "policy_version": state["policy"]["policy_version"],
    }


@app.post("/predict", response_model=Respuesta)
def predict(req: Solicitud) -> Respuesta:
    row = req.model_dump()
    # Misma transformación que en el dataset: proporción préstamo / ingreso con 2 decimales
    row["loan_percent_income"] = round(row["loan_amnt"] / row["person_income"], 2)
    low, high = PLAUSIBLE_RANGES["loan_percent_income"]
    if not low <= row["loan_percent_income"] <= high:
        raise HTTPException(status_code=422, detail="loan_amnt / person_income fuera de rango")
    df = add_dsti(pd.DataFrame([row]))
    model = state["model"]
    columns = list(getattr(model, "feature_names_in_", FEATURES))
    pd_model = float(model.predict_proba(df[columns])[0, 1])

    policy = state["policy"]
    # Margen de conservadurismo: la PD que decide es la del modelo más un margen relativo
    pd_value = min(pd_model * (1 + policy["moc"]), 1.0)
    dsti = float(df["dsti_ratio"].iloc[0])
    decision, motivo = decision_for(pd_value, dsti, req.loan_intent, policy)
    resp = Respuesta(
        pd=round(pd_value, 4),
        pd_modelo=round(pd_model, 4),
        dsti=round(dsti, 4),
        # El rating se asigna con la PD del modelo, la misma con la que se calibraron las bandas
        rating=rating_for(pd_model, policy["rating_bands"]),
        decision=decision,
        motivo=motivo,
        model_version=state["metrics"]["model_version"],
        app_version=APP_VERSION,
    )
    # Registro de cada predicción para la monitorización posterior
    log.info(json.dumps({"event": "prediction", **resp.model_dump(), "intent": req.loan_intent}))
    return resp
