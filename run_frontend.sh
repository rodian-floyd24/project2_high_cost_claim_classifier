#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
streamlit run frontend/app.py --server.port 8501 --server.headless true
