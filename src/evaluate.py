
"""Evaluation utilities for Kestrel service routing."""

from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

TEAM_NAMES = [
    "Repairs",
    "Billing",
    "Filters & Consumables",
    "Installs & Demo",
    "Product Advice",
    "Returns & Replacement",
    "Warranty Claims",
]

FEATURE_COLUMNS = [
    "channel",
    "product_family",
    "warranty_status",
    "request_text",
    "source",
]


def chronological_split(
    data: pd.DataFrame,
    validation_fraction: float = 0.20,
):
    """Split by creation date, keeping newer records for validation."""

    if not 0.05 <= validation_fraction <= 0.40:
        raise ValueError(
            "validation_fraction must be between 0.05 and 0.40."
        )

    if data["created_at_ist"].isna().any():
        raise ValueError(
            "Invalid creation timestamps found. Resolve them before splitting."
        )

    ordered = data.sort_values("created_at_ist").reset_index(drop=True)
    unique_dates = ordered["created_at_ist"].drop_duplicates().sort_values()

    cutoff_index = int(len(unique_dates) * (1 - validation_fraction))

    if cutoff_index <= 0 or cutoff_index >= len(unique_dates):
        raise ValueError(
            "Not enough distinct timestamps for a chronological split."
        )

    cutoff = unique_dates.iloc[cutoff_index]

    train = ordered[ordered["created_at_ist"] < cutoff].copy()
    validation = ordered[ordered["created_at_ist"] >= cutoff].copy()

    if train.empty or validation.empty:
        raise ValueError(
            "Chronological split produced an empty partition."
        )

    return train, validation, cutoff


def calculate_metrics(
    y_true,
    y_pred,
    labels=None,
) -> dict:
    """Calculate metrics for held-out predictions."""

    return {
        "sample_size": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
    }


def save_json_report(report: dict, filename: str) -> Path:
    """Save a JSON report to the reports directory."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return output_path  
       
    
def save_validation_predictions(
    validation: pd.DataFrame,
    y_pred,
    filename: str = "validation_predictions.csv",
    target_column: str = "team_label",
) -> Path:
    """Save held-out predictions with the target used for evaluation."""

    if target_column not in validation.columns:
        raise ValueError(
            f"Validation data does not contain target column "
            f"{target_column!r}."
        )

    if len(validation) != len(y_pred):
        raise ValueError(
            "Validation rows and predictions have different lengths."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output = pd.DataFrame({
        "request_id": validation["request_id"].astype(str).values,
        "target_column": target_column,
        "actual_team": validation[target_column].values,
        "predicted_team": y_pred,
    })

    output["correct"] = (
        output["actual_team"] == output["predicted_team"]
    )

    path = REPORTS_DIR / filename
    output.to_csv(path, index=False)

    return path
