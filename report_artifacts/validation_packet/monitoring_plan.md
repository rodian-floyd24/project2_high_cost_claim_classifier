# Monitoring Plan

## Data Quality

Block publication on duplicate `bene_id` and `year`, missing keys, missing required columns, invalid chronic count, negative costs, or unreasonable enrollment months.

## Drift Thresholds

| Metric | Stable | Warning | Review Required |
|---|---|---|---|
| PSI | `< 0.10` | `0.10` to `< 0.25` | `>= 0.25` |
| Absolute calibration gap | `< 0.02` | `0.02` to `< 0.05` | `>= 0.05` |
| Top-10 capture drop | `< 10%` relative | `10%` to `20%` relative | `> 20%` relative |

## Owners

Analytics owns metric production. Actuarial review owns acceptance decisions. Operations owns queue volume and override monitoring.

## Suspension Triggers

Suspend model promotion or production use when blocking data-quality checks fail, calibration collapses, top-k capture materially drops, or reason-code review identifies implausible drivers.
