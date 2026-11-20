"""Variables calculadas, compartidas por el entrenamiento y la API."""

import pandas as pd

from src.schema import DSTI_ANNUAL_RATE, DSTI_TERM_MONTHS


def add_dsti(df: pd.DataFrame) -> pd.DataFrame:
    """DSTI = cuota mensual simulada / ingreso mensual.

    La cuota usa amortización francesa con un tipo fijo (DSTI_ANNUAL_RATE) y no el tipo real del
    préstamo, porque ese tipo no se conoce antes de conceder el crédito (fuga de datos).
    """
    rate = DSTI_ANNUAL_RATE / 12
    n = DSTI_TERM_MONTHS
    payment = df["loan_amnt"] * rate / (1 - (1 + rate) ** (-n))
    return df.assign(dsti_ratio=payment / (df["person_income"] / 12))
