from __future__ import annotations

ALLOWED_AGE_BANDS = {"under_65", "65_74", "75_84", "85_plus", "unknown"}
ALLOWED_SEX = {"male", "female", "unknown"}
ALLOWED_PRIOR_INTERVENTIONS = {"none", "recent_low_touch", "recent_intensive"}

MAX_REASONABLE_COST = 250_000.0
MAX_REASONABLE_CLAIMS = 1_000


def validate_profile_values(values: dict[str, object]) -> dict[str, object]:
    cost_fields = [
        "rx_total_cost",
        "inpatient_total_cost",
        "outpatient_total_cost",
        "carrier_total_cost",
        "carrier_allowed_total",
        "rx_patient_pay_total",
        "prior_year_annual_claim_cost",
    ]
    count_fields = [
        "inpatient_claim_count",
        "outpatient_claim_count",
        "carrier_claim_count",
        "pde_claim_count",
        "outpatient_ed_claim_count",
        "outpatient_line_count",
        "carrier_line_count",
        "rx_days_supply",
        "total_claim_days",
        "unique_provider_count",
        "prior_year_inpatient_claim_count",
        "prior_year_total_claim_count",
    ]

    for field in cost_fields:
        if float(values.get(field, 0.0) or 0.0) < 0.0:
            raise ValueError(f"{field} must be nonnegative")
    for field in count_fields:
        if int(values.get(field, 0) or 0) < 0:
            raise ValueError(f"{field} must be nonnegative")
    for key, value in values.items():
        if value is None or isinstance(value, bool):
            continue
        if key.endswith("_count") and float(value) < 0:
            raise ValueError(f"{key} must be nonnegative")
        if "cost" in key and float(value) < 0:
            raise ValueError(f"{key} must be nonnegative")
    chronic_count = int(values.get("chronic_condition_count", 0) or 0)
    chronic_flags = [
        bool(values.get(field, False))
        for field in [
            "alzheimers_flag",
            "chf_flag",
            "chronic_kidney_disease_flag",
            "cancer_flag",
            "copd_flag",
            "depression_flag",
            "diabetes_flag",
            "ischemic_heart_disease_flag",
            "osteoporosis_flag",
            "rheumatoid_arthritis_oa_flag",
            "stroke_tia_flag",
        ]
    ]
    if sum(chronic_flags) > chronic_count:
        raise ValueError("chronic_condition_count cannot be less than the number of active chronic flags")
    return values


def input_review_flags(feature_row: dict[str, object]) -> list[str]:
    flags: list[str] = []
    if int(feature_row.get("enrollment_months_count", 0)) <= 0:
        flags.append("zero_enrollment")
    if int(feature_row.get("enrollment_months_count", 0)) < 12:
        flags.append("partial_year_enrollment")
    if float(feature_row.get("annualized_cost_per_enrolled_month", 0.0)) > MAX_REASONABLE_COST:
        flags.append("extreme_annualized_cost")
    if int(feature_row.get("total_claim_count", 0)) > MAX_REASONABLE_CLAIMS:
        flags.append("extreme_claim_count")
    if float(feature_row.get("provider_fragmentation_index", 0.0)) > 1.0:
        flags.append("provider_count_exceeds_claim_count")
    return flags
