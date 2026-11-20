import json

from fastapi.testclient import TestClient

import src.api as api
from src.schema import PLAUSIBLE_RANGES

VALID = {
    "person_age": 30,
    "person_income": 45_000,
    "person_home_ownership": "RENT",
    "person_emp_length": 4,
    "loan_intent": "EDUCATION",
    "loan_amnt": 8_000,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_version"] == "test"


def test_predict_ok(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["pd"] <= 1
    assert body["rating"] in {"R1", "R2", "R3", "R4", "R5"}
    assert body["decision"] in {"APROBAR", "REVISAR", "RECHAZAR"}
    assert body["pd"] >= body["pd_modelo"]  # el margen de conservadurismo nunca baja la PD
    assert body["motivo"]


def test_emp_length_optional(client):
    payload = {k: v for k, v in VALID.items() if k != "person_emp_length"}
    assert client.post("/predict", json=payload).status_code == 200


def test_rejects_extreme_age(client):
    assert client.post("/predict", json={**VALID, "person_age": 144}).status_code == 422


def test_rejects_unknown_category(client):
    assert client.post("/predict", json={**VALID, "loan_intent": "CASINO"}).status_code == 422


def test_rejects_missing_field(client):
    payload = {k: v for k, v in VALID.items() if k != "person_income"}
    assert client.post("/predict", json=payload).status_code == 422


def test_override_hardens_one_intent(client):
    # Simula un hotfix: rechazar todo en VENTURE sin tocar las demás finalidades
    policy = json.loads(client.policy_path.read_text())
    policy["overrides"] = {"VENTURE": {"approve_below": 0.0, "reject_from": 0.0}}
    client.policy_path.write_text(json.dumps(policy))
    with TestClient(api.app) as c2:
        venture = c2.post("/predict", json={**VALID, "loan_intent": "VENTURE"}).json()
        assert venture["decision"] == "RECHAZAR"


def test_rating_bands_order():
    bands = [{"rating": "R2", "max_pd": 0.5}, {"rating": "R1", "max_pd": 0.1}]
    assert api.rating_for(0.05, bands) == "R1"
    assert api.rating_for(0.3, bands) == "R2"


def test_limits_come_from_schema(client):
    low, high = PLAUSIBLE_RANGES["person_age"]
    assert client.post("/predict", json={**VALID, "person_age": high}).status_code == 200
    assert client.post("/predict", json={**VALID, "person_age": high + 1}).status_code == 422
    assert client.post("/predict", json={**VALID, "person_age": low - 1}).status_code == 422


def test_rejects_loan_above_income(client):
    payload = {**VALID, "person_income": 10_000, "loan_amnt": 20_000}
    assert client.post("/predict", json=payload).status_code == 422


def test_rejects_when_dsti_too_high(client):
    # Cuota a 36 meses de 19 000 al 10 %: unos 613 al mes sobre 1 667 de ingreso, DSTI 36.8 %
    payload = {**VALID, "person_income": 20_000, "loan_amnt": 19_000}
    body = client.post("/predict", json=payload).json()
    assert body["decision"] == "RECHAZAR"
    assert body["dsti"] > 0.35
    assert "DSTI" in body["motivo"]
