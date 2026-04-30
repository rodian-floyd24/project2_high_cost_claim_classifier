from __future__ import annotations

from databricks.modeling_utils import validate_required_columns


class DummyFrame:
    columns = ["bene_id", "year", "annual_claim_cost"]


def test_required_gold_columns_check() -> None:
    assert validate_required_columns(DummyFrame(), ["bene_id", "year"]) == []
    assert validate_required_columns(DummyFrame(), ["bene_id", "target_year"]) == ["target_year"]
