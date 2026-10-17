
"""Train and compare models for historical labels and resolution outcomes."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import (
    VALID_TEAMS,
    load_datasets,
    load_reference_files,
    normalize_team_label,
)
from src.evaluate import (
    FEATURE_COLUMNS,
    TEAM_NAMES,
    chronological_split,
    calculate_metrics,
    save_json_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CATEGORICAL_COLUMNS = [
    "channel",
    "product_family",
    "warranty_status",
    "source",
]

OUTCOME_TARGET = "outcome_team"


def build_feature_transformer(include_structured: bool) -> ColumnTransformer:
    """Build text features, optionally including categorical metadata."""

    transformers = [
        (
            "text",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=80000,
                sublinear_tf=True,
            ),
            "request_text",
        )
    ]

    if include_structured:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_candidates() -> dict:
    """Create reproducible candidate models."""

    return {
        "majority_baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "text_logistic_regression": Pipeline([
            (
                "features",
                build_feature_transformer(
                    include_structured=False
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]),
        "text_plus_metadata": Pipeline([
            (
                "features",
                build_feature_transformer(
                    include_structured=True
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]),
        "text_plus_metadata_balanced": Pipeline([
            (
                "features",
                build_feature_transformer(
                    include_structured=True
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),
    }


def attach_resolution_outcomes(
    data: pd.DataFrame,
    resolution: pd.DataFrame,
) -> pd.DataFrame:
    """Join final resolution teams without using outcomes as model features."""

    required = {
        "request_id",
        "first_team",
        "final_team",
        "transfers",
    }
    missing = required - set(resolution.columns)

    if missing:
        raise ValueError(
            f"Resolution log is missing columns: {sorted(missing)}"
        )

    if resolution["request_id"].duplicated().any():
        raise ValueError("Resolution log contains duplicate request IDs.")

    outcomes = resolution[
        ["request_id", "first_team", "final_team", "transfers"]
    ].copy()

    if outcomes[["first_team", "final_team"]].isna().any().any():
        raise ValueError("Resolution log contains missing team names.")

    outcomes["first_team"] = outcomes["first_team"].map(
        normalize_team_label
    )
    outcomes[OUTCOME_TARGET] = outcomes["final_team"].map(
        normalize_team_label
    )

    outcomes = outcomes.drop(columns=["final_team"])

    merged = data.merge(
        outcomes,
        on="request_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((merged["_merge"] != "both").sum())
    merged = merged.drop(columns="_merge")

    if unmatched:
        raise ValueError(
            f"{unmatched} training requests have no resolution record."
        )

    if len(merged) != len(data):
        raise ValueError("Joining resolution outcomes changed row count.")

    if not set(merged[OUTCOME_TARGET]).issubset(VALID_TEAMS):
        raise ValueError("Unexpected team names in resolution outcomes.")

    return merged


def evaluate_target(
    train_part: pd.DataFrame,
    validation_part: pd.DataFrame,
    target: str,
) -> tuple[dict, dict, dict]:
    """Evaluate every candidate against one target on the same time split."""

    X_train = train_part[FEATURE_COLUMNS]
    y_train = train_part[target]

    X_val = validation_part[FEATURE_COLUMNS]
    y_val = validation_part[target]

    results = {}
    predictions = {}
    fitted_models = {}

    for name, model in build_candidates().items():
        print(f"  Training {name} for target {target}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = calculate_metrics(
            y_val,
            y_pred,
            labels=TEAM_NAMES,
        )

        results[name] = metrics
        predictions[name] = y_pred
        fitted_models[name] = model

        print(
            f"    Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f}"
        )

    best_name = max(
        results,
        key=lambda name: (
            results[name]["accuracy"],
            results[name]["macro_f1"],
        ),
    )

    return results, predictions, fitted_models


def main() -> None:
    """Compare both targets, select the outcome model, and save evidence."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training and reference data...")
    data, _ = load_datasets()
    _, resolution = load_reference_files()
    data = attach_resolution_outcomes(data, resolution)

    for target in ["team_label", OUTCOME_TARGET]:
        missing_teams = set(TEAM_NAMES) - set(data[target].unique())
        if missing_teams:
            raise ValueError(
                f"{target} is missing expected teams: "
                f"{sorted(missing_teams)}"
            )

    train_part, validation_part, cutoff = chronological_split(
        data,
        validation_fraction=0.20,
    )

    if (
        train_part["created_at_ist"].max()
        >= validation_part["created_at_ist"].min()
    ):
        raise ValueError("Chronological split is invalid.")

    print("\nChronological validation split")
    print("-" * 50)
    print(f"Training rows:     {len(train_part):,}")
    print(f"Validation rows:   {len(validation_part):,}")
    print(f"Validation starts: {cutoff}")
    print(
        "Training ends:     "
        f"{train_part['created_at_ist'].max()}"
    )

    all_results = {}
    all_predictions = {}
    all_models = {}
    comparison_rows = []

    for target in ["team_label", OUTCOME_TARGET]:
        print(f"\nEvaluating target: {target}")

        results, predictions, models = evaluate_target(
            train_part,
            validation_part,
            target,
        )

        all_results[target] = results
        all_predictions[target] = predictions
        all_models[target] = models

        for name, metrics in results.items():
            comparison_rows.append({
                "target": target,
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "validation_rows": metrics["sample_size"],
            })

    outcome_results = all_results[OUTCOME_TARGET]
    outcome_predictions = all_predictions[OUTCOME_TARGET]
    outcome_models = all_models[OUTCOME_TARGET]

    best_name = max(
        outcome_results,
        key=lambda name: (
            outcome_results[name]["accuracy"],
            outcome_results[name]["macro_f1"],
        ),
    )

    best_metrics = outcome_results[best_name]
    best_predictions = outcome_predictions[best_name]

    # Compare the existing bot's first team with the actual final team.
    bot_metrics = calculate_metrics(
        validation_part[OUTCOME_TARGET],
        validation_part["first_team"],
        labels=TEAM_NAMES,
    )

    historical_best_name = max(
        all_results["team_label"],
        key=lambda name: (
            all_results["team_label"][name]["accuracy"],
            all_results["team_label"][name]["macro_f1"],
        ),
    )
    historical_best_metrics = all_results["team_label"][
        historical_best_name
    ]

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(
        f"Existing bot vs final team: "
        f"{bot_metrics['accuracy']:.4f}"
    )
    print(
        f"Best historical-label model ({historical_best_name}): "
        f"{historical_best_metrics['accuracy']:.4f}"
    )
    print(
        f"Selected outcome model ({best_name}): "
        f"{best_metrics['accuracy']:.4f}"
    )
    print(
        "Outcome improvement over existing bot: "
        f"{best_metrics['accuracy'] - bot_metrics['accuracy']:+.4f}"
    )

    # Save a combined model comparison table.
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    pd.DataFrame(comparison_rows).sort_values(
        ["target", "accuracy", "macro_f1"],
        ascending=[True, False, False],
    ).to_csv(comparison_path, index=False)

    # Save the selected outcome model's confusion matrix.
    pd.DataFrame(
        best_metrics["confusion_matrix"],
        index=TEAM_NAMES,
        columns=TEAM_NAMES,
    ).rename_axis("actual_team").to_csv(
        REPORTS_DIR / "confusion_matrix.csv"
    )

    # Save validation predictions with both reference labels.
    validation_output = pd.DataFrame({
        "request_id": validation_part["request_id"].astype(str).values,
        "historical_label": validation_part["team_label"].values,
        "existing_bot_team": validation_part["first_team"].values,
        "actual_final_team": validation_part[OUTCOME_TARGET].values,
        "predicted_team": best_predictions,
    })
    validation_output["correct_vs_final_team"] = (
        validation_output["actual_final_team"]
        == validation_output["predicted_team"]
    )
    validation_output["existing_bot_correct_vs_final_team"] = (
        validation_output["actual_final_team"]
        == validation_output["existing_bot_team"]
    )
    validation_output.to_csv(
        REPORTS_DIR / "validation_predictions.csv",
        index=False,
    )

    report = {
        "validation_method": (
            "Chronological holdout by created_at_ist; "
            "same split for both targets"
        ),
        "validation_cutoff": str(cutoff),
        "training_rows": int(len(train_part)),
        "validation_rows": int(len(validation_part)),
        "selected_target": OUTCOME_TARGET,
        "selection_rule": (
            "Highest final-team accuracy, then highest macro-F1"
        ),
        "candidate_results_by_target": all_results,
        "selected_model": best_name,
        "selected_validation_accuracy": best_metrics["accuracy"],
        "selected_validation_macro_f1": best_metrics["macro_f1"],
        "selected_validation_error_rate": 1 - best_metrics["accuracy"],
        "existing_bot_final_team_accuracy": bot_metrics["accuracy"],
        "historical_label_best_model": historical_best_name,
        "historical_label_best_accuracy": (
            historical_best_metrics["accuracy"]
        ),
        "outcome_accuracy_gain_over_existing_bot": (
            best_metrics["accuracy"] - bot_metrics["accuracy"]
        ),
        "team_names": TEAM_NAMES,
        "limitations": [
            "Retrospective validation is not a live pilot.",
            "Historical final teams may reflect existing workflows.",
            "Accuracy does not establish causal cost savings.",
            "Model confidence thresholds have not been calibrated here.",
        ],
    }
    save_json_report(report, "validation_metrics.json")

    # Refit the selected outcome model on all available labelled requests.
    print("\nRetraining selected outcome model on all training rows...")
    final_model = build_candidates()[best_name]
    final_model.fit(
        data[FEATURE_COLUMNS],
        data[OUTCOME_TARGET],
    )

    model_path = MODEL_DIR / "router.joblib"
    joblib.dump(
        {
            "pipeline": final_model,
            "model_name": best_name,
            "feature_columns": FEATURE_COLUMNS,
            "team_names": TEAM_NAMES,
            "target_column": OUTCOME_TARGET,
            "validation_metrics": {
                "accuracy": best_metrics["accuracy"],
                "macro_f1": best_metrics["macro_f1"],
                "error_rate": 1 - best_metrics["accuracy"],
                "existing_bot_final_team_accuracy": (
                    bot_metrics["accuracy"]
                ),
                "historical_label_best_accuracy": (
                    historical_best_metrics["accuracy"]
                ),
            },
            "training_rows": int(len(data)),
        },
        model_path,
    )

    print(f"\nSaved model: {model_path}")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved report: {REPORTS_DIR / 'validation_metrics.json'}")
    print(
        "Saved validation predictions: "
        f"{REPORTS_DIR / 'validation_predictions.csv'}"
    )
    print("\nTraining and validation completed.")


if __name__ == "__main__":
    main()
