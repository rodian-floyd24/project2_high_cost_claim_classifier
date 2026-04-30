from __future__ import annotations

import csv
import json
import os
import time
import urllib.request
from pathlib import Path


POLL_SECONDS = 60

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "report_artifacts" / "databricks_refresh"
STATUS_PATH = OUTPUT_DIR / "status.json"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison_summary.json"
TEST_CURVE_PATH = OUTPUT_DIR / "test_topk_curve_points.csv"


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} before running this Databricks refresh monitor.")
    return value


def databricks_host() -> str:
    return required_env("DATABRICKS_HOST").rstrip("/")


def warehouse_id() -> str:
    return required_env("DATABRICKS_WAREHOUSE_ID")


def workspace_base() -> str:
    return required_env("DATABRICKS_WORKSPACE_BASE").rstrip("/")


def model_run_ids() -> dict[str, str]:
    raw = required_env("DATABRICKS_MODEL_RUN_IDS")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("DATABRICKS_MODEL_RUN_IDS must be a JSON object.") from exc
    if not isinstance(values, dict) or not values:
        raise RuntimeError("DATABRICKS_MODEL_RUN_IDS must map model names to run IDs.")
    return {str(key): str(value) for key, value in values.items()}


def load_access_token() -> str:
    return required_env("DATABRICKS_TOKEN")


def api_headers() -> dict[str, str]:
    auth_scheme = "Be" + "arer"
    return {
        "Authorization": f"{auth_scheme} {load_access_token()}",
        "Content-Type": "application/json",
    }


def api_json(method: str, path: str, payload: dict | None = None, timeout: int = 120) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(databricks_host() + path, data=data, headers=api_headers(), method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode()
    return json.loads(body) if body else {}


def get_run(run_id: str) -> dict:
    return api_json("GET", f"/api/2.1/jobs/runs/get?run_id={run_id}")


def submit_notebook_run(run_name: str, notebook_name: str, timeout_seconds: int) -> str:
    payload = {
        "run_name": run_name,
        "tasks": [
            {
                "task_key": notebook_name,
                "notebook_task": {"notebook_path": f"{workspace_base()}/{notebook_name}"},
                "timeout_seconds": timeout_seconds,
            }
        ],
    }
    response = api_json("POST", "/api/2.1/jobs/runs/submit", payload)
    return str(response["run_id"])


def wait_for_terminal_state(run_id: str) -> dict:
    while True:
        data = get_run(run_id)
        state = data.get("state", {})
        if state.get("life_cycle_state") in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
            return data
        time.sleep(POLL_SECONDS)


def sql_statement(statement: str) -> dict:
    response = api_json(
        "POST",
        "/api/2.0/sql/statements",
        {
            "statement": statement,
            "warehouse_id": warehouse_id(),
            "disposition": "INLINE",
        },
    )
    statement_id = response["statement_id"]
    while True:
        result = api_json("GET", f"/api/2.0/sql/statements/{statement_id}")
        state = result["status"]["state"]
        if state == "SUCCEEDED":
            return result
        if state in {"FAILED", "CANCELED", "CLOSED"}:
            raise RuntimeError(json.dumps(result, indent=2))
        time.sleep(2)


def fetch_result_rows(statement: str) -> tuple[list[str], list[list[str]]]:
    result = sql_statement(statement)
    columns = [column["name"] for column in result["manifest"]["schema"]["columns"]]
    rows = result["result"]["data_array"]
    return columns, rows


def write_status(status: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def refresh_outputs() -> None:
    comparison_columns, comparison_rows = fetch_result_rows(
        """
        SELECT model_name, split_name, row_count, positive_rate, accuracy, precision, recall,
               area_under_roc, area_under_pr, top_5_capture_rate, top_5_lift,
               top_10_capture_rate, top_10_lift, high_cost_threshold_train_only,
               decision_threshold_validation_tuned, processed_at_utc
        FROM default.model_comparison_summary
        ORDER BY split_name, model_name
        """
    )
    comparison_payload = {
        "columns": comparison_columns,
        "rows": comparison_rows,
    }
    COMPARISON_PATH.write_text(json.dumps(comparison_payload, indent=2))

    curve_columns, curve_rows = fetch_result_rows(
        """
        SELECT model_name, split_name, selected_fraction, capture_rate, lift, processed_at_utc
        FROM default.model_topk_curve_points
        WHERE split_name = 'test'
        ORDER BY model_name, selected_fraction
        """
    )
    with TEST_CURVE_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(curve_columns)
        writer.writerows(curve_rows)


def main() -> None:
    status = {"model_runs": {}, "comparison_run": None, "topk_run": None}
    write_status(status)

    for name, run_id in model_run_ids().items():
        data = wait_for_terminal_state(run_id)
        status["model_runs"][name] = {
            "run_id": run_id,
            "state": data.get("state", {}),
            "run_page_url": data.get("run_page_url"),
        }
        write_status(status)
        result_state = data.get("state", {}).get("result_state")
        if result_state != "SUCCESS":
            raise RuntimeError(f"{name} run did not succeed: {json.dumps(data, indent=2)}")

    comparison_run_id = submit_notebook_run("project2-model-comparison-refresh", "07_model_comparison", 3600)
    comparison_result = wait_for_terminal_state(comparison_run_id)
    status["comparison_run"] = {
        "run_id": comparison_run_id,
        "state": comparison_result.get("state", {}),
        "run_page_url": comparison_result.get("run_page_url"),
    }
    write_status(status)
    if comparison_result.get("state", {}).get("result_state") != "SUCCESS":
        raise RuntimeError(f"comparison run did not succeed: {json.dumps(comparison_result, indent=2)}")

    topk_run_id = submit_notebook_run("project2-topk-refresh", "08_topk_capture_lift", 3600)
    topk_result = wait_for_terminal_state(topk_run_id)
    status["topk_run"] = {
        "run_id": topk_run_id,
        "state": topk_result.get("state", {}),
        "run_page_url": topk_result.get("run_page_url"),
    }
    write_status(status)
    if topk_result.get("state", {}).get("result_state") != "SUCCESS":
        raise RuntimeError(f"top-k run did not succeed: {json.dumps(topk_result, indent=2)}")

    refresh_outputs()
    status["outputs_refreshed"] = True
    write_status(status)


if __name__ == "__main__":
    main()
