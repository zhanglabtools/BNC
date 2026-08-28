from __future__ import annotations

from pathlib import Path

from bnc_repro.validation import validate_all


def test_supplied_reference_data_inventory() -> None:
    root = Path(__file__).resolve().parents[1]
    results = validate_all(root / "paper_data")
    assert len(results) == 6
    assert all(result["status"] for result in results)

