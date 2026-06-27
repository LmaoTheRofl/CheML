import pytest

from chemx.domains import load_domain
from chemx.models import Prediction, SourceRef
from chemx.validation import deduplicate_prediction, validate_prediction


def make_prediction() -> Prediction:
    spec = load_domain("eyedrops")
    values = {field.name: None for field in spec.fields}
    evidence = {field.name: [] for field in spec.fields}
    return Prediction(domain=spec.slug, records=[{"values": values, "evidence": evidence}])


def test_prediction_matches_domain_contract() -> None:
    prediction = make_prediction()
    assert validate_prediction(prediction, load_domain("eyedrops")) is prediction


def test_prediction_rejects_missing_field() -> None:
    prediction = make_prediction()
    prediction.records[0].values.pop("smiles")
    with pytest.raises(ValueError, match="missing"):
        validate_prediction(prediction, load_domain("eyedrops"))


def test_prediction_rejects_wrong_scalar_type() -> None:
    prediction = make_prediction()
    prediction.records[0].values["PMID"] = "fast"
    with pytest.raises(ValueError, match="must be number"):
        validate_prediction(prediction, load_domain("eyedrops"))


def test_exact_duplicate_is_removed_but_distinct_evidence_is_preserved() -> None:
    prediction = make_prediction()
    prediction.records.append(prediction.records[0].model_copy(deep=True))
    assert len(deduplicate_prediction(prediction).records) == 1
    prediction.records[1].evidence["smiles"] = [SourceRef(page=1, kind="figure")]
    assert len(deduplicate_prediction(prediction).records) == 2
