
"""Analyze routing errors for the outcome-based model.

Creates aggregate diagnostic reports without changing the production model
or predictions.csv. Customer message text is not exported.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data import (
    load_datasets,
    load_reference_files,
    normalize_team_label,
)
from src.evaluate import FEATURE_COLUMNS, TEAM_NAMES, chronological_split
from src.train import build_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, _ = load_datasets()
    _, resolution = load_reference_files()

    resolution["request_id"] = resolution["request_id"].astype(str)
    resolution["outcome_team"] = resolution["final_team"].map(
        normalize_team_label
    )

    train["request_id"] = train["request_id"].astype(str)

    data = train.merge(
        resolution[["request_id", "outcome_team"]],
        on="request_id",
        how="left",
        validate="one_to_one",
    )

    if data["outcome_team"].isna().any():
        raise ValueError("Missing outcome labels after joining resolution data.")

    train_part, val_part, cutoff = chronological_split(
        data,
        validation_fraction=0.20,
    )

    X_train = train_part[FEATURE_COLUMNS]
    X_val = val_part[FEATURE_COLUMNS]

    y_train = train_part["outcome_team"]
    y_true = val_part["outcome_team"].reset_index(drop=True)
    bot_pred = val_part["team_label"].reset_index(drop=True)

    # The experiment showed this is the best outcome-based candidate.
    model = build_candidates()["text_logistic_regression"]
    model.fit(X_train, y_train)

    model_pred = pd.Series(model.predict(X_val)).reset_index(drop=True)

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Validation start: {cutoff}")
    print(f"Validation requests: {len(y_true):,}")
    print(f"Old bot outcome accuracy: {(bot_pred == y_true).mean():.2%}")
    print(f"Outcome model accuracy:   {(model_pred == y_true).mean():.2%}")

    # Team-level metrics for both approaches.
    bot_report = classification_report(
        y_true,
        bot_pred,
        labels=TEAM_NAMES,
        output_dict=True,
        zero_division=0,
    )
    model_report = classification_report(
        y_true,
        model_pred,
        labels=TEAM_NAMES,
        output_dict=True,
        zero_division=0,
    )

    team_rows = []

    for team in TEAM_NAMES:
        team_rows.append({
            "team": team,
            "actual_validation_requests": int((y_true == team).sum()),
            "bot_precision": bot_report[team]["precision"],
            "bot_recall": bot_report[team]["recall"],
            "bot_f1": bot_report[team]["f1-score"],
            "model_precision": model_report[team]["precision"],
            "model_recall": model_report[team]["recall"],
            "model_f1": model_report[team]["f1-score"],
        })

    team_table = pd.DataFrame(team_rows)
    team_path = REPORTS_DIR / "outcome_team_metrics.csv"
    team_table.to_csv(team_path, index=False)

    # Confusion matrices: rows are actual outcomes, columns are predictions.
    bot_cm = confusion_matrix(y_true, bot_pred, labels=TEAM_NAMES)
    model_cm = confusion_matrix(y_true, model_pred, labels=TEAM_NAMES)

    pd.DataFrame(
        bot_cm,
        index=TEAM_NAMES,
        columns=TEAM_NAMES,
    ).rename_axis("actual_team").to_csv(
        REPORTS_DIR / "outcome_bot_confusion_matrix.csv"
    )

    pd.DataFrame(
        model_cm,
        index=TEAM_NAMES,
        columns=TEAM_NAMES,
    ).rename_axis("actual_team").to_csv(
        REPORTS_DIR / "outcome_model_confusion_matrix.csv"
    )

    # Count specific mistakes, excluding correct predictions.
    error_rows = []

    for actual, predicted in zip(y_true, model_pred):
        if actual != predicted:
            error_rows.append({
                "actual_team": actual,
                "predicted_team": predicted,
            })

    if error_rows:
        errors = pd.DataFrame(error_rows)
        top_errors = (
            errors.groupby(["actual_team", "predicted_team"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    else:
        top_errors = pd.DataFrame(
            columns=["actual_team", "predicted_team", "count"]
        )

    error_path = REPORTS_DIR / "outcome_model_top_errors.csv"
    top_errors.to_csv(error_path, index=False)

    print("\n=== TEAM-LEVEL METRICS ===")
    print(team_table.round(3).to_string(index=False))

    print("\n=== TOP 15 MODEL CONFUSIONS ===")
    if top_errors.empty:
        print("No validation errors.")
    else:
        print(top_errors.head(15).to_string(index=False))

    print("\nSaved aggregate reports:")
    print(team_path)
    print(REPORTS_DIR / "outcome_bot_confusion_matrix.csv")
    print(REPORTS_DIR / "outcome_model_confusion_matrix.csv")
    print(error_path)
    print("\nNo customer request text was exported.")


if __name__ == "__main__":
    main()
