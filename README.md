# Scoring de crédito · Modelos de servicio A2 · Grupo 1

API de scoring (FastAPI) con CI/CD en GitHub Actions, imagen Docker en GHCR y flujo Git-Flow.

## Estructura (frente MLOps)

| Ruta | Qué es |
|---|---|
| `src/api.py` | API: `POST /predict` (PD, rating, decisión) y `GET /health` |
| `src/schema.py` | Contrato de columnas, categorías y rangos |
| `src/validate_data.py` | Validación de calidad del dato (falla el CI si no se cumple) |
| `src/train.py` | Entrenamiento basado en `Credit_Score_Model_2_0.ipynb`: XGBoost (campeón) y regresión logística (referencia) |
| `src/features.py` | Cálculo de `dsti_ratio`, compartido por entrenamiento y API |
| `src/gate.py` | Puerta de calidad: Gini mínimo y comparación con el campeón |
| `config/policy.json` | Bandas de rating y cortes de decisión (valores de relleno) |
| `loadtest/load_test.py` | Prueba de carga en flujo continuo y comprobación del SLA |
| `.github/workflows/ci.yml` | Pipeline completo |
| `.github/workflows/rollback.yml` | Vuelta manual de producción a una versión estable |

## Contrato con el Data Scientist

`train.py` debe dejar en `model/`:
- `model.joblib`: `Pipeline` de scikit-learn que recibe un `DataFrame` con `FEATURES` (incluido `dsti_ratio`).
- `metrics.json`: como mínimo `model_version`, `auc`, `gini`, `ks`, `baseline` y `features`.

## Ejecutar en local (PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m src.validate_data data/credit_risk_dataset.csv
python -m src.train --version 1.0.0
python -m src.gate
pytest -q
uvicorn src.api:app --port 8000
# en otra terminal
python loadtest/load_test.py --rps 50 --duration 60
```

## Docker

```powershell
docker build --build-arg APP_VERSION=1.0.0 -t credit-scoring:1.0.0 .
docker run -p 8000:8000 credit-scoring:1.0.0
```
