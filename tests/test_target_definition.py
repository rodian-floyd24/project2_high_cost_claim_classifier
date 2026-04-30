from __future__ import annotations

import pytest

from databricks.modeling_utils import reject_target_leakage


def test_reject_target_leakage_blocks_target_columns() -> None:
    with pytest.raises(ValueError):
        reject_target_leakage(["age_years", "target_annual_claim_cost", "label"])


def test_reject_target_leakage_allows_prior_year_features() -> None:
    reject_target_leakage(["age_years", "prior_year_annual_claim_cost", "current_year_high_cost_indicator"])
