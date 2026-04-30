#!/usr/bin/env bash
set -euo pipefail

python3 -m compileall -q backend databricks scripts tests
python3 -m pytest -q
python3 scripts/check_no_leakage.py
