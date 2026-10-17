
"""Analyze held-out routing errors using local assessment data."""

from pathlib import Path

import pandas as pd

from src.data import TRAIN_FILE, load_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    predictions_path = REPORTS_DIR / "validation_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing {predictions_path}. Run python -m src.train first."
        )

    train = load_csv(TRAIN_FILE)
    predictions = pd.read_csv(predictions_path, dtype={"request_id": str})

    train["request_id"] = train["request_id"].astype(str)

    merged = predictions.merge(
        train[
            [
                "request_id",
                "created_at_ist",
                "channel",
                "product_family",
                "warranty_status",
                "request_text",
                "source",
            ]
        ],
        on="request_id",
        how="left",
        validate="one_to_one",
    )

    if merged["request_text"].isna().any():
        raise ValueError(
            "Some validation predictions could not be matched to training rows."
        )

    errors = merged.loc[~merged["correct"]].copy()

    print("\n" + "=" * 65)
    print("KESTREL ROUTING — VALIDATION ERROR ANALYSIS")
    print("=" * 65)

    print(f"Validation records: {len(merged):,}")
    print(f"Correct predictions: {int(merged['correct'].sum()):,}")
    print(f"Incorrect predictions: {len(errors):,}")
    print(f"Observed error rate: {len(errors) / len(merged):.2%}")

    print("\nMost common actual -> predicted mistakes:")
    if errors.empty:
        print("No errors found.")
    else:
        pairs = (
            errors.groupby(["actual_team", "predicted_team"])
            .size()
            .sort_values(ascending=False)
            .rename("count")
            .reset_index()
        )
        print(pairs.head(15).to_string(index=False))

    print("\nError count by actual team:")
    if errors.empty:
        print("No errors found.")
    else:
        print(
            errors["actual_team"]
            .value_counts()
            .to_string()
        )

    print("\nError rate by channel:")
    print(
        merged.groupby("channel", dropna=False)["correct"]
        .agg(records="size", accuracy="mean")
        .assign(error_rate=lambda x: 1 - x["accuracy"])
        .sort_values("error_rate", ascending=False)
        .to_string()
    )

    print("\nError rate by source:")
    print(
        merged.groupby("source", dropna=False)["correct"]
        .agg(records="size", accuracy="mean")
        .assign(error_rate=lambda x: 1 - x["accuracy"])
        .sort_values("error_rate", ascending=False)
        .to_string()
    )

    # This file contains request-level operational data.
    # Keep it local; do not publish it to a public repository.
    output_path = REPORTS_DIR / "error_analysis.csv"
    errors.to_csv(output_path, index=False)

    print(f"\nSaved detailed errors locally to: {output_path}")
    print("\nInspect these examples before changing the model.")


if __name__ == "__main__":
    main()
