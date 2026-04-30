from __future__ import annotations


def top_risk_drivers(feature_row: dict[str, object], max_drivers: int = 5) -> list[str]:
    drivers: list[tuple[int, str]] = []
    chronic_count = int(feature_row.get("chronic_condition_count", 0))
    total_claim_count = int(feature_row.get("total_claim_count", 0))
    inpatient_claim_count = int(feature_row.get("inpatient_claim_count", 0))
    claims_per_month = float(feature_row.get("claims_per_enrollment_month", 0.0))
    provider_count = int(feature_row.get("unique_provider_count", 0))
    prior_high_cost = int(feature_row.get("prior_year_high_cost_indicator", 0))
    cost_per_month = float(feature_row.get("cost_per_enrollment_month", 0.0))
    enrollment_months = int(feature_row.get("enrollment_months_count", 0))

    if chronic_count >= 6:
        drivers.append((95, "high chronic condition count"))
    elif chronic_count >= 3:
        drivers.append((70, "moderate chronic condition burden"))
    if inpatient_claim_count > 0:
        drivers.append((90, "prior inpatient utilization"))
    if claims_per_month >= 2.0 or total_claim_count >= 20:
        drivers.append((85, "high claims per enrollment month"))
    if provider_count >= 6:
        drivers.append((75, "high provider fragmentation"))
    if prior_high_cost:
        drivers.append((80, "prior high-cost year indicator"))
    if cost_per_month >= 1000.0:
        drivers.append((65, "high prior-year cost intensity"))
    if 0 < enrollment_months < 12:
        drivers.append((50, "partial-year enrollment requires review"))

    if not drivers:
        drivers.append((10, "routine utilization and chronic burden profile"))
    return [driver for _, driver in sorted(drivers, reverse=True)[:max_drivers]]


def reason_code_version() -> str:
    return "reason_codes_v1"
