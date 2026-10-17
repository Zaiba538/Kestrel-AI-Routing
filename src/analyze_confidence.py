
"""Evaluate confidence-based routing on chronological validation data."""

import pandas as pd

from src.data import load_datasets, load_reference_files
from src.evaluate import FEATURE_COLUMNS, chronological_split
from src.train import (
    OUTCOME_TARGET,
    attach_resolution_outcomes,
    build_candidates,
)

CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
MARGIN_THRESHOLD = 0.10


def main():
    data, _ = load_datasets()
    _, resolution = load_reference_files()
    data = attach_resolution_outcomes(data, resolution)

    train_part, validation_part, cutoff = chronological_split(
        data,
        validation_fraction=0.20,
    )

    model = build_candidates()["text_logistic_regression"]
    model.fit(
        train_part[FEATURE_COLUMNS],
        train_part[OUTCOME_TARGET],
    )

    probabilities = model.predict_proba(
        validation_part[FEATURE_COLUMNS]
    )
    classes = list(model.named_steps["classifier"].classes_)

    predicted_indices = probabilities.argmax(axis=1)
    predicted_teams = [classes[i] for i in predicted_indices]
    confidence = probabilities.max(axis=1)

    sorted_probabilities = pd.DataFrame(
        probabilities
    ).apply(
        lambda row: sorted(row, reverse=True),
        axis=1,
    )

    margins = sorted_probabilities.map(
        lambda values: (
            values[0] - values[1] if len(values) >= 2 else 1.0
        )
    )

    actual = validation_part[OUTCOME_TARGET].reset_index(drop=True)
    predicted = pd.Series(predicted_teams)
    confidence = pd.Series(confidence)
    margins = pd.Series(margins)

    print(f"Validation cutoff: {cutoff}")
    print(f"Validation rows: {len(validation_part):,}")
    print(f"Overall accuracy: {(actual == predicted).mean():.4%}")

    print("\nConfidence threshold results")
    print("-" * 72)
    print(
        f"{'Threshold':>10} {'Auto-routed':>12} "
        f"{'Coverage':>10} {'Accuracy':>10} {'Errors':>8}"
    )

    for threshold in CONFIDENCE_THRESHOLDS:
        selected = confidence >= threshold
        count = int(selected.sum())
        coverage = float(selected.mean())
        accuracy = (
            float((actual[selected] == predicted[selected]).mean())
            if count else float("nan")
        )
        errors = (
            int((actual[selected] != predicted[selected]).sum())
            if count else 0
        )

        print(
            f"{threshold:>9.0%} {count:>12,} "
            f"{coverage:>9.2%} {accuracy:>9.2%} {errors:>8,}"
        )

    # Apply the current policy: confidence >= 0.60 and margin >= 0.10.
    auto_route = (
        (confidence >= 0.60)
        & (margins >= MARGIN_THRESHOLD)
    )
    count = int(auto_route.sum())
    coverage = float(auto_route.mean())
    accuracy = (
        float((actual[auto_route] == predicted[auto_route]).mean())
        if count else float("nan")
    )

    print("\nCurrent policy result")
    print("-" * 40)
    print(f"Auto-routed requests: {count:,} / {len(actual):,}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Accuracy among auto-routed: {accuracy:.2%}")
    print(f"Sent for review: {len(actual) - count:,}")


if __name__ == "__main__":
    main()
