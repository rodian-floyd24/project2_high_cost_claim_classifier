"""Register raw CMS synthetic claims files as bronze Delta tables."""

from __future__ import annotations

import json
import os
import re
from uuid import uuid4

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T


DEFAULT_INVENTORY = os.environ.get(
    "CMS_INVENTORY_PATH",
    "/Volumes/workspace/default/project2_high_cost_claim_classifier/planning/cms_source_inventory.json",
)
DEFAULT_STORAGE_ROOT = os.environ.get(
    "OBJECT_STORAGE_ROOT",
    "/Volumes/workspace/default/project2_high_cost_claim_classifier/object_storage",
)
DEFAULT_DATABASE = os.environ.get("BRONZE_DATABASE", "default")
AUDIT_TABLE_NAME = "bronze_audit_summary"

TABLE_NAMES = {
    "beneficiary_summary": "bronze_beneficiary_summary",
    "inpatient_claims": "bronze_inpatient_claims",
    "outpatient_claims": "bronze_outpatient_claims",
    "carrier_claims": "bronze_carrier_claims",
    "prescription_drug_events": "bronze_pde",
}


def load_inventory(path: str) -> dict:
    text = "".join(row.value for row in spark.read.text(path).collect())
    return json.loads(text)


def grouped_entries(inventory: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for entry in inventory["files"]:
        grouped.setdefault(entry["entity"], []).append(entry)
    return grouped


def entity_paths(storage_root: str, entity: str) -> tuple[str, str]:
    entity_root = f"{storage_root}/bronze/{entity}"
    return f"{entity_root}/raw", f"{entity_root}/extracted"


def list_file_names(dbfs_path: str) -> list[str]:
    try:
        return sorted(item.name for item in dbutils.fs.ls(dbfs_path) if not item.isDir())
    except Exception:
        return []


def parse_entry_year(entry: dict) -> int | None:
    for key in ("logical_name", "expected_extracted_name", "preferred_href_substring", "display_name"):
        value = entry.get(key)
        if not value:
            continue
        match = re.search(r"(20\d{2})", value)
        if match:
            return int(match.group(1))
    return None


def source_year_column(entity: str, entries: list[dict], source_file: F.Column) -> F.Column:
    if entity != "beneficiary_summary":
        return F.lit(None).cast("int")

    year_expr = F.lit(None).cast("int")
    for entry in entries:
        entry_year = parse_entry_year(entry)
        extracted_name = entry.get("expected_extracted_name")
        if entry_year and extracted_name:
            year_expr = F.when(source_file.contains(extracted_name), F.lit(entry_year)).otherwise(year_expr)
    return year_expr


def read_entity_raw_files(entity: str, entries: list[dict], default_read_options: dict) -> DataFrame:
    _, extracted_dir = entity_paths(DEFAULT_STORAGE_ROOT, entity)
    paths = [file_info.path for file_info in dbutils.fs.ls(extracted_dir) if not file_info.isDir()]
    if not paths:
        raise FileNotFoundError(f"No extracted files found for entity '{entity}' at {extracted_dir}")

    reader = spark.read
    fmt = default_read_options.get("format", "csv")
    if fmt != "csv":
        raise ValueError(f"Unsupported bronze read format: {fmt}")

    reader = reader.option("header", str(default_read_options.get("header", True)).lower())
    reader = reader.option("sep", default_read_options.get("sep", ","))
    df = reader.csv(paths)
    source_file = F.col("_metadata.file_path")

    expected_join_key = next((entry.get("expected_join_key") for entry in entries if entry.get("expected_join_key")), None)
    return (
        df.withColumn("_bronze_entity", F.lit(entity))
        .withColumn("_bronze_loaded_at_utc", F.current_timestamp())
        .withColumn("_bronze_source_file", source_file)
        .withColumn("_bronze_source_year", source_year_column(entity, entries, source_file))
        .withColumn("_expected_join_key", F.lit(expected_join_key))
    )


def write_bronze_table(df: DataFrame, database: str, table_name: str, mode: str = "overwrite") -> None:
    full_name = f"{database}.{table_name}"
    (
        df.write.format("delta")
        .mode(mode)
        .option("overwriteSchema", "true" if mode == "overwrite" else "false")
        .option("mergeSchema", "true" if mode == "append" else "false")
        .saveAsTable(full_name)
    )


def validate_entity(df: DataFrame, entries: list[dict]) -> dict[str, int | bool | None]:
    expected_row_count = sum(entry.get("expected_row_count", 0) for entry in entries)
    has_join_key = "DESYNPUF_ID" in df.columns
    row_count = df.count()
    null_desynpuf_id_count = None
    distinct_desynpuf_id_count = None
    if has_join_key:
        stats = df.agg(
            F.sum(F.when(F.col("DESYNPUF_ID").isNull(), 1).otherwise(0)).alias("null_desynpuf_id_count"),
            F.countDistinct("DESYNPUF_ID").alias("distinct_desynpuf_id_count"),
        ).collect()[0]
        null_desynpuf_id_count = int(stats["null_desynpuf_id_count"])
        distinct_desynpuf_id_count = int(stats["distinct_desynpuf_id_count"])
    return {
        "row_count": row_count,
        "expected_row_count": expected_row_count,
        "has_desynpuf_id": has_join_key,
        "null_desynpuf_id_count": null_desynpuf_id_count,
        "distinct_desynpuf_id_count": distinct_desynpuf_id_count,
    }


def build_audit_rows(
    run_id: str,
    entity: str,
    table_name: str,
    raw_file_names: list[str],
    extracted_file_names: list[str],
    validation: dict[str, int | bool | None],
    status: str,
) -> list[dict]:
    max_len = max(len(raw_file_names), len(extracted_file_names), 1)
    rows = []
    for index in range(max_len):
        rows.append(
            {
                "run_id": run_id,
                "entity_name": entity,
                "raw_file_name": raw_file_names[index] if index < len(raw_file_names) else None,
                "extracted_file_name": extracted_file_names[index] if index < len(extracted_file_names) else None,
                "landed_row_count": validation["row_count"],
                "expected_row_count": validation["expected_row_count"],
                "null_desynpuf_id_count": validation["null_desynpuf_id_count"],
                "distinct_desynpuf_id_count": validation["distinct_desynpuf_id_count"],
                "ingestion_timestamp_utc": None,
                "bronze_table_name": table_name,
                "status": status,
            }
        )
    return rows


def create_audit_dataframe(rows: list[dict]) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("run_id", T.StringType(), False),
            T.StructField("entity_name", T.StringType(), False),
            T.StructField("raw_file_name", T.StringType(), True),
            T.StructField("extracted_file_name", T.StringType(), True),
            T.StructField("landed_row_count", T.LongType(), True),
            T.StructField("expected_row_count", T.LongType(), True),
            T.StructField("null_desynpuf_id_count", T.LongType(), True),
            T.StructField("distinct_desynpuf_id_count", T.LongType(), True),
            T.StructField("ingestion_timestamp_utc", T.TimestampType(), True),
            T.StructField("bronze_table_name", T.StringType(), False),
            T.StructField("status", T.StringType(), False),
        ]
    )
    return spark.createDataFrame(rows, schema=schema).withColumn(
        "ingestion_timestamp_utc", F.current_timestamp()
    )


def main() -> None:
    inventory = load_inventory(DEFAULT_INVENTORY)
    database = DEFAULT_DATABASE
    entities = grouped_entries(inventory)
    default_read_options = inventory.get("default_read_options", {})
    run_id = uuid4().hex

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
    summaries = []
    audit_rows = []

    for entity, entries in entities.items():
        raw_dir, extracted_dir = entity_paths(DEFAULT_STORAGE_ROOT, entity)
        raw_file_names = list_file_names(raw_dir)
        extracted_file_names = list_file_names(extracted_dir)
        df = read_entity_raw_files(entity, entries, default_read_options)
        table_name = TABLE_NAMES[entity]
        write_bronze_table(df, database, table_name)
        validation = validate_entity(df, entries)
        full_table_name = f"{database}.{table_name}"
        summaries.append({"entity": entity, "table": full_table_name, **validation})
        audit_rows.extend(
            build_audit_rows(
                run_id=run_id,
                entity=entity,
                table_name=full_table_name,
                raw_file_names=raw_file_names,
                extracted_file_names=extracted_file_names,
                validation=validation,
                status="ok" if validation["has_desynpuf_id"] else "missing_desynpuf_id",
            )
        )

    audit_df = create_audit_dataframe(audit_rows)
    write_bronze_table(audit_df, database, AUDIT_TABLE_NAME, mode="append")

    for summary in summaries:
        print(
            f"{summary['entity']}: table={summary['table']} row_count={summary['row_count']} "
            f"expected_row_count={summary['expected_row_count']} "
            f"has_desynpuf_id={summary['has_desynpuf_id']} "
            f"null_desynpuf_id_count={summary['null_desynpuf_id_count']} "
            f"distinct_desynpuf_id_count={summary['distinct_desynpuf_id_count']}"
        )
    print(f"bronze audit summary written to {database}.{AUDIT_TABLE_NAME} for run_id={run_id}")


if __name__ == "__main__":
    main()
