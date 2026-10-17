
"""
Kestrel Home — Batch Routing Predictions.

Loads the trained routing model, predicts all unlabelled
requests, and generates predictions.csv in the repository root.

Submission format:
    request_id,team
"""

from pathlib import Path

import joblib
import pandas as pd

from src.data import (
    PROJECT_ROOT,
    TEST_FILE,
    VALID_TEAMS,
    load_csv,
    load_datasets,
)

MODEL_PATH = PROJECT_ROOT / "models" / "router.joblib"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_submission.csv"
OUTPUT_PATH = PROJECT_ROOT / "predictions.csv"


def predict_batch() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("KESTREL HOME — BATCH PREDICTION")
    print("=" * 60)

    # Load validated assessment data.
    _, test = load_datasets()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run python -m src.train first."
        )

    # Load the model bundle created by src/train.py.
    bundle = joblib.load(MODEL_PATH)

    model = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    missing_features = set(feature_columns) - set(test.columns)

    if missing_features:
        raise ValueError(
            f"Missing test features: {sorted(missing_features)}"
        )

    print(f"\nModel: {bundle['model_name']}")
    print(f"Test requests: {len(test):,}")

    # Predict with the same feature columns used during training.
    X_test = test[feature_columns]
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "request_id": test["request_id"].astype(str).values,
        "team": predictions,
    })

    # Validate the required submission schema.
    expected_columns = ["request_id", "team"]

    if submission.columns.tolist() != expected_columns:
        raise ValueError("Incorrect submission columns.")

    if len(submission) != len(test):
        raise ValueError("Prediction row count mismatch.")

    if submission["request_id"].isna().any():
        raise ValueError("Missing request IDs in submission.")

    if submission["request_id"].duplicated().any():
        raise ValueError("Duplicate request IDs in submission.")

    if submission["team"].isna().any():
        raise ValueError("Missing predicted team labels.")

    invalid_teams = set(submission["team"]) - VALID_TEAMS

    if invalid_teams:
        raise ValueError(
            f"Invalid predicted teams: {sorted(invalid_teams)}"
        )

    # Check against the supplied submission template.
    sample = load_csv(SAMPLE_PATH)

    if sample.columns.tolist() != expected_columns:
        raise ValueError(
            "Sample submission schema differs from expected format."
        )

    if len(sample) != len(submission):
        raise ValueError(
            f"Template contains {len(sample)} rows but predictions "
            f"contain {len(submission)} rows."
        )

    sample_ids = sample["request_id"].astype(str)
    prediction_ids = submission["request_id"].astype(str)

    if sample_ids.duplicated().any():
        raise ValueError("Duplicate request IDs in sample submission.")

    if set(sample_ids) != set(prediction_ids):
        raise ValueError(
            "Prediction request IDs do not match the sample submission."
        )

    # Preserve the exact request order in the official template.
    submission = (
        sample[["request_id"]]
        .assign(request_id=lambda df: df["request_id"].astype(str))
        .merge(
            submission,
            on="request_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if submission["team"].isna().any():
        raise ValueError("Some template requests have no prediction.")

    # Save only the two required columns in the repository root.
    submission.to_csv(OUTPUT_PATH, index=False)

    print("\nPrediction distribution:")
    print(submission["team"].value_counts().to_string())

    print("\nSubmission checks:")
    print(f"Rows: {len(submission):,}")
    print(f"Columns: {submission.columns.tolist()}")
    print(f"Unique request IDs: {submission['request_id'].nunique():,}")
    print(f"Valid teams: {not submission['team'].isna().any()}")

    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nBatch prediction completed successfully.")
    print("=" * 60)

    return submission


if __name__ == "__main__":
    predict_batch()
