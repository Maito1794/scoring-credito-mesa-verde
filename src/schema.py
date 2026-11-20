"""Contrato de datos compartido por la validación, el entrenamiento y la API."""

TARGET = "loan_status"

# Columnas tal como vienen en credit_risk_dataset.csv
RAW_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

CATEGORIES = {
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT",
    ],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["Y", "N"],
}

# Variables del modelo (Data Scientist). Se excluyen loan_grade y loan_int_rate por fuga de datos:
# en la realidad no se conocen antes de conceder el préstamo
NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "dsti_ratio",
]
CATEGORICAL_FEATURES = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Variables que se calculan (no vienen en el CSV ni en la solicitud)
DERIVED_FEATURES = ["dsti_ratio"]

# Cuota simulada para el DSTI: amortización francesa a 36 meses con tipo fijo del 10 %
DSTI_TERM_MONTHS = 36
DSTI_ANNUAL_RATE = 0.10

# Rangos plausibles: la limpieza descarta y la API rechaza lo que quede fuera
PLAUSIBLE_RANGES = {
    "person_age": (18, 100),
    "person_emp_length": (0, 60),
    "person_income": (1, 10_000_000),
    "loan_amnt": (1, 1_000_000),
    "loan_int_rate": (0, 100),
    "loan_percent_income": (0, 1),
    "cb_person_cred_hist_length": (0, 60),
}
