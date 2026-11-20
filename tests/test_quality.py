from src.gate import evaluate
from src.validate_data import validate


def test_validation_passes_on_clean_data(synthetic_df):
    assert validate(synthetic_df)["ok"]


def test_validation_fails_on_missing_column(synthetic_df):
    report = validate(synthetic_df.drop(columns=["loan_intent"]))
    assert not report["ok"]


def test_validation_fails_on_unknown_category(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "person_home_ownership"] = "CASTLE"
    assert not validate(df)["ok"]


def test_validation_fails_on_too_many_nulls(synthetic_df):
    df = synthetic_df.copy()
    df.loc[: len(df) // 2, "loan_int_rate"] = None
    assert not validate(df)["ok"]


def test_validation_only_warns_on_outliers(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "person_age"] = 144
    report = validate(df)
    assert report["ok"] and any("person_age" in w for w in report["warnings"])


def test_gate_pass_and_fail():
    assert evaluate({"gini": 0.55}, 0.40, None, 0.03) == []
    assert evaluate({"gini": 0.35}, 0.40, None, 0.03)
    assert evaluate({"gini": 0.50}, 0.40, {"gini": 0.56}, 0.03)  # pierde 6 puntos vs campeón


def test_gate_requires_beating_baseline():
    base = {"algorithm": "logistic", "gini": 0.70}
    assert evaluate({"gini": 0.80, "algorithm": "xgboost", "baseline": base}, 0.4, None, 0.03) == []
    assert evaluate({"gini": 0.65, "algorithm": "xgboost", "baseline": base}, 0.4, None, 0.03)
