from chemx.normalize import NOT_DETECTED, multiset_metric, normalize_number, normalize_value


def test_numeric_normalization_happy_path() -> None:
    assert normalize_number(" 1,250 ") == "1.25"


def test_missing_values_normalize_to_not_detected() -> None:
    assert normalize_value(None) == NOT_DETECTED
    assert normalize_value(float("nan")) == NOT_DETECTED
    assert normalize_value("ND") == NOT_DETECTED


def test_malformed_numeric_value_is_preserved() -> None:
    assert normalize_number("< 0,05") == "<0.05"


def test_extreme_numeric_sizes_are_not_rounded() -> None:
    assert normalize_number("0,0000000000001") == "0.0000000000001"
    assert normalize_number("100000000000000000000") == "100000000000000000000"


def test_multiset_metric_counts_duplicates() -> None:
    metric = multiset_metric(["a", "a", "b"], ["a", "b", "b"])
    assert metric.true_positive == 2
    assert metric.precision == 2 / 3
    assert metric.recall == 2 / 3


def test_empty_multisets_are_perfect_match() -> None:
    metric = multiset_metric([], [])
    assert metric.precision == metric.recall == metric.f1 == 1.0

