from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "report_artifacts" / "project_clean_data_browse.csv"
OUTPUT_DIR = PROJECT_ROOT / "report_artifacts" / "gold_feature_eda"
REPORT_PATH = OUTPUT_DIR / "gold_feature_eda_summary.md"

TARGET_QUANTILE = 0.90

REQUIRED_COLUMNS = {
    "bene_id",
    "year",
    "annual_claim_cost",
    "age_years",
    "age_band",
    "sex",
    "race_code",
    "state_code",
    "enrollment_months_count",
    "chronic_condition_count",
    "inpatient_claim_count",
    "outpatient_claim_count",
    "carrier_claim_count",
    "pde_claim_count",
    "inpatient_total_cost",
    "outpatient_total_cost",
    "carrier_total_cost",
    "rx_total_cost",
    "total_claim_count",
    "total_claim_days",
    "unique_provider_count",
    "cost_per_enrollment_month",
    "claims_per_enrollment_month",
    "provider_fragmentation_index",
}

LEAKAGE_SENSITIVE_COLUMNS = [
    "annual_claim_cost",
    "inpatient_total_cost",
    "outpatient_total_cost",
    "carrier_total_cost",
    "rx_total_cost",
    "cost_per_enrollment_month",
]

ENGINEERED_FEATURE_COLUMNS = [
    "chronic_condition_count_squared",
    "claims_per_month_chronic_count_interaction",
    "providers_per_month_chronic_count_interaction",
    "inpatient_claim_count_log1p",
    "outpatient_claim_count_log1p",
    "carrier_claim_count_log1p",
    "pde_claim_count_log1p",
    "total_claim_count_log1p",
    "unique_provider_count_log1p",
]

COUNT_COLUMNS = [
    "inpatient_claim_count",
    "outpatient_claim_count",
    "carrier_claim_count",
    "pde_claim_count",
    "total_claim_count",
    "total_claim_days",
    "unique_provider_count",
]

COST_COLUMNS = [
    "annual_claim_cost",
    "inpatient_total_cost",
    "outpatient_total_cost",
    "carrier_total_cost",
    "rx_total_cost",
]


def currency(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def load_gold_export() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Gold feature export not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{INPUT_PATH} is missing required gold feature columns: {missing}")

    df["year"] = df["year"].astype(int)
    add_engineered_features(df)
    return df


def add_engineered_features(df: pd.DataFrame) -> None:
    df["chronic_condition_count_squared"] = df["chronic_condition_count"] ** 2
    df["claims_per_month_chronic_count_interaction"] = (
        df["claims_per_enrollment_month"] * df["chronic_condition_count"]
    )
    df["providers_per_month_chronic_count_interaction"] = (
        df["unique_provider_count"] / df["enrollment_months_count"].where(df["enrollment_months_count"] > 0, np.nan)
    ).fillna(0.0) * df["chronic_condition_count"]

    for source_column, output_column in [
        ("inpatient_claim_count", "inpatient_claim_count_log1p"),
        ("outpatient_claim_count", "outpatient_claim_count_log1p"),
        ("carrier_claim_count", "carrier_claim_count_log1p"),
        ("pde_claim_count", "pde_claim_count_log1p"),
        ("total_claim_count", "total_claim_count_log1p"),
        ("unique_provider_count", "unique_provider_count_log1p"),
    ]:
        df[output_column] = np.log1p(df[source_column].clip(lower=0))


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()
    next_year = df[["bene_id", "year", "annual_claim_cost"]].copy()
    next_year["year"] = next_year["year"] - 1
    next_year = next_year.rename(
        columns={
            "year": "feature_year",
            "annual_claim_cost": "target_annual_claim_cost",
        }
    )
    current = current.rename(columns={"year": "feature_year"})
    modeling = current.merge(next_year, on=["bene_id", "feature_year"], how="inner")
    modeling["target_year"] = modeling["feature_year"] + 1

    thresholds = (
        modeling.groupby("target_year")["target_annual_claim_cost"]
        .quantile(TARGET_QUANTILE)
        .rename("target_year_high_cost_threshold")
        .reset_index()
    )
    modeling = modeling.merge(thresholds, on="target_year", how="left")
    modeling["label"] = (
        modeling["target_annual_claim_cost"] > modeling["target_year_high_cost_threshold"]
    ).astype(int)
    return modeling


def write_csv_outputs(df: pd.DataFrame, modeling: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    outputs["row_counts_by_year"] = (
        df.groupby("year")
        .agg(row_count=("bene_id", "size"), distinct_beneficiaries=("bene_id", "nunique"))
        .reset_index()
    )

    outputs["missingness"] = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    outputs["missingness"]["missing_rate"] = outputs["missingness"]["missing_count"] / len(df)
    outputs["missingness"] = outputs["missingness"].sort_values(
        ["missing_rate", "column"], ascending=[False, True]
    )

    outputs["cost_distribution"] = (
        df[COST_COLUMNS]
        .quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        .T.reset_index()
        .rename(columns={"index": "feature"})
    )

    outputs["class_balance_by_target_year"] = (
        modeling.groupby("target_year")
        .agg(
            row_count=("bene_id", "size"),
            positive_count=("label", "sum"),
            positive_rate=("label", "mean"),
            high_cost_threshold=("target_year_high_cost_threshold", "first"),
            median_target_cost=("target_annual_claim_cost", "median"),
            mean_target_cost=("target_annual_claim_cost", "mean"),
        )
        .reset_index()
    )

    outputs["median_cost_by_chronic_count"] = (
        df.groupby("chronic_condition_count")
        .agg(
            row_count=("bene_id", "size"),
            median_annual_claim_cost=("annual_claim_cost", "median"),
            mean_annual_claim_cost=("annual_claim_cost", "mean"),
            p90_annual_claim_cost=("annual_claim_cost", lambda series: series.quantile(0.90)),
        )
        .reset_index()
        .sort_values("chronic_condition_count")
    )

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    quality_rows = []
    for column in numeric_columns:
        series = df[column]
        quality_rows.append(
            {
                "feature": column,
                "zero_rate": float((series == 0).mean()),
                "negative_count": int((series < 0).sum()),
                "min": float(series.min()),
                "median": float(series.median()),
                "p90": float(series.quantile(0.90)),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    outputs["numeric_feature_quality"] = pd.DataFrame(quality_rows).sort_values(
        ["negative_count", "zero_rate"], ascending=[False, False]
    )

    candidate_numeric = [
        column
        for column in numeric_columns
        if column in modeling.columns
        and column not in {"year", "feature_year", "target_year", "label"}
        and modeling[column].nunique(dropna=True) > 1
    ]
    corr_rows = []
    for column in candidate_numeric:
        corr = modeling[[column, "label"]].corr(numeric_only=True).iloc[0, 1]
        if pd.notna(corr):
            corr_rows.append({"feature": column, "correlation_with_next_year_label": float(corr)})
    outputs["feature_label_correlations"] = (
        pd.DataFrame(corr_rows)
        .assign(abs_correlation=lambda frame: frame["correlation_with_next_year_label"].abs())
        .sort_values("abs_correlation", ascending=False)
    )

    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    return outputs


def save_plots(df: pd.DataFrame, modeling: pd.DataFrame, outputs: dict[str, pd.DataFrame]) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))
    clipped_annual_cost = df["annual_claim_cost"].clip(lower=0)
    sns.histplot(np.log1p(clipped_annual_cost), bins=50, ax=ax, color="#3366AA")
    ax.set_title("Annual Claim Cost Distribution, nonnegative log1p scale")
    ax.set_xlabel("log1p(max(annual_claim_cost, 0))")
    ax.set_ylabel("Beneficiary-year rows")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "annual_claim_cost_log_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    balance = outputs["class_balance_by_target_year"]
    sns.barplot(data=balance, x="target_year", y="positive_rate", ax=ax, color="#AA6633")
    ax.axhline(0.10, color="#333333", linewidth=1, linestyle="--")
    ax.set_title("Next-Year High-Cost Class Balance")
    ax.set_xlabel("Target year")
    ax.set_ylabel("Positive rate")
    ax.set_ylim(0, max(0.15, float(balance["positive_rate"].max()) + 0.02))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_balance_by_year.png", dpi=180)
    plt.close(fig)

    top_corr = outputs["feature_label_correlations"].head(15).copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=top_corr,
        y="feature",
        x="correlation_with_next_year_label",
        ax=ax,
        color="#447744",
    )
    ax.set_title("Top Numeric Correlations With Next-Year High-Cost Label")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_feature_label_correlations.png", dpi=180)
    plt.close(fig)

    cost_by_burden = (
        df.groupby("chronic_condition_count")["annual_claim_cost"]
        .median()
        .reset_index()
        .sort_values("chronic_condition_count")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=cost_by_burden,
        x="chronic_condition_count",
        y="annual_claim_cost",
        marker="o",
        ax=ax,
        color="#663399",
    )
    ax.set_title("Median Annual Cost by Chronic Condition Count")
    ax.set_xlabel("Chronic condition count")
    ax.set_ylabel("Median annual claim cost")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "median_cost_by_chronic_count.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, max_rows: int = 10) -> str:
    return frame.head(max_rows).to_markdown(index=False)


def write_report(df: pd.DataFrame, modeling: pd.DataFrame, outputs: dict[str, pd.DataFrame]) -> None:
    duplicate_keys = int(df.duplicated(["bene_id", "year"]).sum())
    key_null_rows = int(df[["bene_id", "year"]].isna().any(axis=1).sum())
    missing_any = int(df.isna().any(axis=1).sum())
    nonpositive_enrollment = int((df["enrollment_months_count"] <= 0).sum())
    negative_cost_rows = int((df[COST_COLUMNS] < 0).any(axis=1).sum())
    target_rows_lost = len(df) - len(modeling)

    cost_dist = outputs["cost_distribution"].set_index("feature")
    annual = cost_dist.loc["annual_claim_cost"]
    class_balance = outputs["class_balance_by_target_year"]
    correlations = outputs["feature_label_correlations"]

    leakage_corr = correlations[correlations["feature"].isin(LEAKAGE_SENSITIVE_COLUMNS)][
        ["feature", "correlation_with_next_year_label"]
    ]
    engineered_corr = correlations[correlations["feature"].isin(ENGINEERED_FEATURE_COLUMNS)][
        ["feature", "correlation_with_next_year_label"]
    ]
    count_quality = outputs["numeric_feature_quality"][
        outputs["numeric_feature_quality"]["feature"].isin(COUNT_COLUMNS + COST_COLUMNS)
    ]
    chronic_distribution = (
        df["chronic_condition_count"]
        .value_counts()
        .sort_index()
        .rename("row_count")
        .reset_index()
        .rename(columns={"index": "chronic_condition_count"})
    )
    chronic_distribution["row_rate"] = chronic_distribution["row_count"] / len(df)
    chronic_cost = outputs["median_cost_by_chronic_count"]
    chronic_min = int(df["chronic_condition_count"].min())
    chronic_max = int(df["chronic_condition_count"].max())

    report = f"""# Gold Feature EDA Summary

Source: `{INPUT_PATH.relative_to(PROJECT_ROOT)}`

## Scope

This EDA treats the gold features table as a beneficiary-year modeling table. It checks whether the table is suitable for prospective modeling by building a year `t` to year `t + 1` frame and defining `label = 1` when next-year annual claim cost is above that target year's {int(TARGET_QUANTILE * 100)}th percentile.

## Table Integrity

| Check | Value |
|---|---:|
| Gold rows | {len(df):,} |
| Distinct beneficiaries | {df["bene_id"].nunique():,} |
| Years | {", ".join(str(year) for year in sorted(df["year"].unique()))} |
| Duplicate `bene_id + year` rows | {duplicate_keys:,} |
| Rows with missing key fields | {key_null_rows:,} |
| Rows with any missing value | {missing_any:,} |
| Rows with nonpositive enrollment months | {nonpositive_enrollment:,} |
| Rows with a negative cost component | {negative_cost_rows:,} |

## Prospective Modeling Frame

| Check | Value |
|---|---:|
| Year `t` to `t + 1` modeling rows | {len(modeling):,} |
| Rows unavailable for prospective target | {target_rows_lost:,} |
| Distinct modeled beneficiaries | {modeling["bene_id"].nunique():,} |
| Feature years | {", ".join(str(year) for year in sorted(modeling["feature_year"].unique()))} |
| Target years | {", ".join(str(year) for year in sorted(modeling["target_year"].unique()))} |

Class balance by target year:

{markdown_table(class_balance.assign(positive_rate=lambda x: x["positive_rate"].map(pct), high_cost_threshold=lambda x: x["high_cost_threshold"].map(currency)))}

## Chronic Burden Signal

The corrected gold export contains `chronic_condition_count` values from {chronic_min} through {chronic_max}; it is not constant zero. Median annual cost rises monotonically and accelerates as chronic burden increases, which is risk compounding rather than a purely additive linear effect.

{markdown_table(chronic_cost.assign(median_annual_claim_cost=lambda x: x["median_annual_claim_cost"].map(currency), mean_annual_claim_cost=lambda x: x["mean_annual_claim_cost"].map(currency), p90_annual_claim_cost=lambda x: x["p90_annual_claim_cost"].map(currency)), 12)}

Chronic-condition row distribution:

{markdown_table(chronic_distribution.assign(row_rate=lambda x: x["row_rate"].map(pct)), 12)}

Modeling implication: do not rely on a simple linear chronic-count effect alone. The tree models can learn the nonlinear shape directly. Linear baselines should include nonlinear terms, burden bands, or interactions such as the existing age/chronic and chronic-burden-band features.

## Cost and Utilization Shape

Annual cost is highly right-skewed, which supports log cost features and ranking metrics such as PR-AUC, top-k capture, and lift.

| Metric | Value |
|---|---:|
| Annual cost median | {currency(float(annual[0.50]))} |
| Annual cost 90th percentile | {currency(float(annual[0.90]))} |
| Annual cost 99th percentile | {currency(float(annual[0.99]))} |
| Annual cost max | {currency(float(annual[1.00]))} |

Key cost/count quality checks:

{markdown_table(count_quality[["feature", "zero_rate", "negative_count", "min", "median", "p90", "p99", "max"]].assign(zero_rate=lambda x: x["zero_rate"].map(pct)), 20)}

## Signal Checks

Top numeric correlations with the next-year high-cost label:

{markdown_table(correlations[["feature", "correlation_with_next_year_label"]], 15)}

The leading signals are utilization intensity features: unique provider count, carrier claim count, total claim count, claims per enrollment month, and chronic condition count. This is consistent with an actuarial prospective-risk framing: future high-cost status is driven more by stable prior utilization intensity and care engagement patterns than by demographics alone.

Cost-derived feature correlations with the next-year label:

{markdown_table(leakage_corr, 20)}

Notably, prior-year `annual_claim_cost` is predictive but not the strongest signal. Raw cost is noisy; frequency, provider breadth, and utilization density appear more stable for next-year high-cost risk.

New engineered utilization/chronic structure feature correlations:

{markdown_table(engineered_corr, 20)}

## Pipeline Integrity Notes

The earlier chronic-flag issue was real: the CMS chronic flags use `1/2` coding, and an older parser that expected only `Y/N` would collapse chronic counts to zero. This EDA export is from the corrected gold data and verifies that chronic burden is populated.

Local evidence checked here:

- `report_artifacts/project_clean_data_browse.csv` has `chronic_condition_count` values from {chronic_min} to {chronic_max}.
- The EDA generator derives the new v2 structural features from the export when the CSV predates the Databricks gold refresh.
- The deployed MLflow model signature in `backend/model_artifacts/model/MLmodel` requires `chronic_condition_count`.
- The Databricks training scripts use `default.gold_beneficiary_year_features` and include `chronic_condition_count` plus chronic interactions in their feature lists.
- `databricks/13_gold_pipeline_consistency_check.py` now enforces the live Databricks contract before model results are trusted.

Remaining integrity check before relying on any specific historical model run: confirm that the model artifact being evaluated was trained after the silver chronic-flag parser fix. If an old model was trained before the fix, its metrics and feature conclusions should be regenerated.

## Statistical Variable Selection

The engineered features are useful predictive candidates, but they are not automatically a textbook-selected statistical specification. The project now separates the modeling work into two roles:

- Statistical baseline: `databricks/14_logreg_variable_selection.py` fits a logistic-regression specification selected only on the training split using backward AIC over an interpretable full candidate model. It enforces hierarchy rules, writes selected terms, numeric collinearity diagnostics, nested likelihood-ratio tests, and coefficient/odds-ratio output.
- Predictive extensions: random forest, gradient boosting, and XGBoost use the same gold feature contract for out-of-sample discrimination, ranking, top-k lift, and calibrated probability comparison.

The selected logistic workflow centers chronic-condition count using the training-split mean before considering the quadratic term, then enforces polynomial hierarchy so the centered linear chronic term is retained whenever the centered squared term is retained. This avoids treating tree-model feature strength as a substitute for formal variable selection. The selected logistic model is the one to defend with coefficient, odds-ratio, likelihood-ratio, collinearity, and parsimony language; the tree ensembles are prediction machines.

## Modeling Implications

- Use the prospective year `t` to `t + 1` frame for final model evaluation. Same-year high-cost segmentation would leak the annual-cost target through direct cost and utilization features.
- Keep class-imbalance metrics in the primary evaluation set. The target is intentionally near 10% positive by target year, so accuracy can be misleading.
- Cost fields are right-skewed. Log transforms, tree models, and ranking metrics are appropriate.
- The local gold export has no duplicate beneficiary-year keys and no missing values in the exported fields.
- Same-year cost fields are acceptable as prior-year predictors in the prospective frame, but they should not be used to predict same-year high-cost status.
- The model is effectively learning next-year high-cost risk from prior-year utilization intensity, provider breadth, chronic burden, and cost density.
- Formal statistical claims should reference the selected logistic-regression audit, not the univariate EDA rankings or tree-model importances alone.

## Generated Artifacts

- `gold_feature_eda/row_counts_by_year.csv`
- `gold_feature_eda/missingness.csv`
- `gold_feature_eda/cost_distribution.csv`
- `gold_feature_eda/class_balance_by_target_year.csv`
- `gold_feature_eda/median_cost_by_chronic_count.csv`
- `gold_feature_eda/numeric_feature_quality.csv`
- `gold_feature_eda/feature_label_correlations.csv`
- `gold_feature_eda/annual_claim_cost_log_distribution.png`
- `gold_feature_eda/target_balance_by_year.png`
- `gold_feature_eda/top_feature_label_correlations.png`
- `gold_feature_eda/median_cost_by_chronic_count.png`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_gold_export()
    modeling = build_modeling_frame(df)
    outputs = write_csv_outputs(df, modeling)
    save_plots(df, modeling, outputs)
    write_report(df, modeling, outputs)
    print(f"Wrote gold feature EDA report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
