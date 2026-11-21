import json
from pathlib import Path

POLICY = json.loads(Path("config/policy.json").read_text(encoding="utf-8"))


def _cuts(extra: dict | None = None) -> dict:
    return {**POLICY["cutoffs"], **(extra or {})}


def test_cutoffs_are_probabilities():
    # Evita errores como escribir 15 (porcentaje) en lugar de 0.15
    for cuts in [_cuts(), *[_cuts(o) for o in POLICY.get("overrides", {}).values()]]:
        for name in ("approve_below", "reject_from", "max_dsti"):
            assert 0 <= cuts[name] <= 1, f"{name}={cuts[name]} debe estar entre 0 y 1"
        assert cuts["approve_below"] <= cuts["reject_from"]


def test_moc_is_a_fraction():
    assert 0 <= POLICY["moc"] < 1


def test_rating_bands_are_increasing_and_end_at_one():
    limits = [b["max_pd"] for b in POLICY["rating_bands"]]
    assert limits == sorted(limits) and len(set(limits)) == len(limits)
    assert limits[-1] == 1.0
