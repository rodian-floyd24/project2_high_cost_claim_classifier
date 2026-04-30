# Deployment Runbook

## Pre-Deployment Checks

Run local tests, leakage checks, schema checks, validation packet generation, and model artifact health checks.

## Smoke Test

Start the backend and call `/health`, `/metadata`, and `/decision_support` with a known valid payload. Confirm the response includes model version, risk score, risk tier, recommended action, reason codes, and manual-review status.

## Rollback

Rollback to the previous locked model artifact if artifact loading fails, score distributions drift materially, calibration checks fail, or operations reports unacceptable queue behavior.
