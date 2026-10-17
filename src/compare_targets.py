
"""Compare historical-label and resolution-outcome routing objectives.

This is a diagnostic experiment. It does not replace the production model
or modify predictions.csv.
"""

from pathlib import Path

import pandas as pd

from src.data import (
    load_datasets,
    load_reference_files,
    normalize_team_label,
)
from src.evaluate import (
    FEATURE_COLUMNS,
    TEAM_NAMES,
    chronological_split,
    calculate_metrics,
)
from src.train import build_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data and resolution outcomes...")
    train, _ = load_datasets()
    _, resolution = load_reference_files()

    # Normalize historical team names to current functional team names.
    resolution["request_id"] = resolution["request_id"].astype(str)
    resolution["outcome_team"] = resolution["final_team"].map(
        normalize_team_label
    )
    train["request_id"] = train["request_id"].astype(str)

    # Keep only fields required to define the outcome.
    outcomes = resolution[["request_id", "outcome_team"]]

    data = train.merge(
        outcomes,
        on="request_id",
        how="left",
        validate="one_to_one",
    )

    if data["outcome_team"].isna().any():
        missing = int(data["outcome_team"].isna().sum())
        raise ValueError(
            f"{missing} training requests have no normalized outcome team."
        )

    unknown = (
        set(data["team_label"]) | set(data["outcome_team"])
    ) - set(TEAM_NAMES)

    if unknown:
        raise ValueError(
            f"Unexpected team names after normalization: {sorted(unknown)}"
        )

    # Both experiments use exactly the same chronological split.
    train_part, val_part, cutoff = chronological_split(
        data,
        validation_fraction=0.20,
    )

    X_train = train_part[FEATURE_COLUMNS]
    X_val = val_part[FEATURE_COLUMNS]

    print("\n=== VALIDATION DESIGN ===")
    print(f"Training rows:   {len(train_part):,}")
    print(f"Validation rows: {len(val_part):,}")
    print(f"Validation starts: {cutoff}")
    print(f"Training ends: {train_part['created_at_ist'].max()}")

    # The old bot is evaluated against the actual resolution outcome.
    bot_accuracy = float(
        (
            val_part["team_label"].reset_index(drop=True)
            == val_part["outcome_team"].reset_index(drop=True)
        ).mean()
    )

    print("\nOld bot agreement with actual resolution team: "
          f"{bot_accuracy:.4f}")

    experiment_results = []
    detail_report = {
        "validation_method": "Chronological 80/20 split by created_at_ist",
        "validation_cutoff": str(cutoff),
        "training_rows": int(len(train_part)),
        "validation_rows": int(len(val_part)),
        "old_bot_accuracy_vs_outcome": bot_accuracy,
        "experiments": {},
    }

    targets = {
        "historical_label": "team_label",
        "resolution_outcome": "outcome_team",
    }

    for experiment_name, target_column in targets.items():
        print(f"\n{'=' * 65}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"Training target: {target_column}")
        print("=" * 65)

        y_train = train_part[target_column]
        y_val = val_part[target_column]

        results_for_target = {}
        candidates = build_candidates()

        for model_name, model in candidates.items():
            print(f"\nTraining: {model_name}")

            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            metrics = calculate_metrics(
                y_val,
                predictions,
                labels=TEAM_NAMES,
            )

            # Also compare each prediction to the final resolution team.
            outcome_accuracy = float(
                (
                    pd.Series(predictions).reset_index(drop=True)
                    == val_part["outcome_team"].reset_index(drop=True)
                ).mean()
            )

            # And measure agreement with the historical bot label.
            historical_agreement = float(
                (
                    pd.Series(predictions).reset_index(drop=True)
                    == val_part["team_label"].reset_index(drop=True)
                ).mean()
            )

            results_for_target[model_name] = {
                "accuracy_vs_training_target": metrics["accuracy"],
                "macro_f1_vs_training_target": metrics["macro_f1"],
                "accuracy_vs_resolution_outcome": outcome_accuracy,
                "agreement_with_historical_label": historical_agreement,
                "sample_size": metrics["sample_size"],
                "confusion_matrix": metrics["confusion_matrix"],
                "classification_report": metrics["classification_report"],
            }

            experiment_results.append({
                "experiment": experiment_name,
                "target": target_column,
                "model": model_name,
                "accuracy_vs_training_target": metrics["accuracy"],
                "macro_f1_vs_training_target": metrics["macro_f1"],
                "accuracy_vs_resolution_outcome": outcome_accuracy,
                "agreement_with_historical_label": historical_agreement,
                "validation_rows": metrics["sample_size"],
            })

            print(
                f"Target accuracy: {metrics['accuracy']:.4f} | "
                f"Target macro-F1: {metrics['macro_f1']:.4f} | "
                f"Outcome accuracy: {outcome_accuracy:.4f} | "
                f"Historical-label agreement: {historical_agreement:.4f}"
            )

        detail_report["experiments"][experiment_name] = results_for_target

    comparison = pd.DataFrame(experiment_results)
    comparison = comparison.sort_values(
        ["experiment", "accuracy_vs_training_target"],
        ascending=[True, False],
    )

    output_csv = REPORTS_DIR / "target_comparison.csv"
    output_json = REPORTS_DIR / "target_comparison.json"

    comparison.to_csv(output_csv, index=False)

    import json

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(detail_report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("SUMMARY: MODEL COMPARISON")
    print("=" * 65)
    print(comparison.to_string(index=False))

    print(f"\nSaved comparison table: {output_csv}")
    print(f"Saved detailed report:  {output_json}")
    print("\nDiagnostic experiment completed.")
    print("Existing model and predictions.csv were not modified.")


if __name__ == "__main__":
    main()
