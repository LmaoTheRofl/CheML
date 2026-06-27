from pathlib import Path

import pytest

from chemx.domains import load_domain
from chemx.models import Prediction
from chemx.validation import repair_and_canonicalize_prediction, validate_prediction


def test_repair_prediction_schema_adds_missing_and_drops_unknown(tmp_path: Path) -> None:
    spec = load_domain("complexes")
    values = {field.name: None for field in spec.fields}
    values.pop("metal")
    values["unexpected"] = "x"
    prediction = Prediction(domain=spec.slug, records=[{"values": values, "evidence": {}}])

    repaired = repair_and_canonicalize_prediction(
        prediction,
        spec,
        tmp_path,
        require_rdkit=False,
    )

    assert "unexpected" not in repaired.records[0].values
    assert "metal" in repaired.records[0].values
    validate_prediction(repaired, spec)
    assert (tmp_path / "schema_diagnostics.json").is_file()


def test_repair_prediction_canonicalizes_smiles_with_rdkit(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    spec = load_domain("complexes")
    values = {field.name: None for field in spec.fields}
    values["SMILES"] = "C(C)O"
    prediction = Prediction(domain=spec.slug, records=[{"values": values, "evidence": {}}])

    repaired = repair_and_canonicalize_prediction(
        prediction,
        spec,
        tmp_path,
        require_rdkit=True,
    )

    assert repaired.records[0].values["SMILES"] == "CCO"
    validate_prediction(repaired, spec)
