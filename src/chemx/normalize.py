from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

NOT_DETECTED = "NOT_DETECTED"
_MISSING = {"", "-", "--", "n/a", "na", "nan", "none", "null", "not detected", "nd"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in _MISSING


def normalize_number(value: Any) -> str:
    if is_missing(value):
        return NOT_DETECTED
    raw = str(value).strip().replace("\u2212", "-").replace(",", ".")
    raw = re.sub(r"\s+", "", raw)
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return raw
    if number == number.to_integral():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def canonicalize_smiles(value: Any) -> str:
    if is_missing(value):
        return NOT_DETECTED
    raw = str(value).strip()
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(raw)
        if molecule is None:
            return raw
        return Chem.MolToSmiles(molecule, canonical=True)
    except ImportError:
        return raw


def normalize_value(value: Any, *, numeric: bool = False, smiles: bool = False) -> str:
    if smiles:
        return canonicalize_smiles(value)
    if numeric:
        return normalize_number(value)
    if is_missing(value):
        return NOT_DETECTED
    return re.sub(r"\s+", " ", str(value).strip())


@dataclass(frozen=True)
class ColumnMetric:
    precision: float
    recall: float
    f1: float
    true_positive: int
    predicted: int
    expected: int


def multiset_metric(predicted: Iterable[str], expected: Iterable[str]) -> ColumnMetric:
    pred_counter = Counter(predicted)
    gold_counter = Counter(expected)
    true_positive = sum((pred_counter & gold_counter).values())
    pred_total = sum(pred_counter.values())
    gold_total = sum(gold_counter.values())
    precision = true_positive / pred_total if pred_total else (1.0 if gold_total == 0 else 0.0)
    recall = true_positive / gold_total if gold_total else (1.0 if pred_total == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ColumnMetric(precision, recall, f1, true_positive, pred_total, gold_total)
