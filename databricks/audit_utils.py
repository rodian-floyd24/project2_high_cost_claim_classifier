from __future__ import annotations


def append_delta_table(df, table_name: str, merge_schema: bool = False) -> None:
    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", str(bool(merge_schema)).lower())
        .saveAsTable(table_name)
    )


def write_training_audit(metrics, table_name: str) -> None:
    append_delta_table(metrics, table_name, merge_schema=True)


def write_prediction_scores(scores, table_name: str) -> None:
    append_delta_table(scores, table_name, merge_schema=True)


def write_topk_curve(points, table_name: str) -> None:
    append_delta_table(points, table_name, merge_schema=True)


def write_failure_audit(run_id: str, status: str, failure_reason: str):
    from datetime import datetime, timezone

    return {
        "run_id": run_id,
        "status": status,
        "failure_reason": failure_reason,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
