
"""Analyze observed routing transitions and transfer burden.

Uses aggregate statistics only. Does not export customer request text.
"""

from pathlib import Path

import pandas as pd

from src.data import (
    load_datasets,
    load_reference_files,
    normalize_team_label,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, _ = load_datasets()
    _, resolution = load_reference_files()

    train["request_id"] = train["request_id"].astype(str)
    resolution["request_id"] = resolution["request_id"].astype(str)

    resolution["final_team_normalized"] = resolution["final_team"].map(
        normalize_team_label
    )
    resolution["first_team_normalized"] = resolution["first_team"].map(
        normalize_team_label
    )

    data = train.merge(
        resolution[
            [
                "request_id",
                "first_team_normalized",
                "final_team_normalized",
                "transfers",
            ]
        ],
        on="request_id",
        how="left",
        validate="one_to_one",
    )

    if data["transfers"].isna().any():
        raise ValueError("Missing transfer counts after joining resolution data.")

    if (data["transfers"] < 0).any():
        raise ValueError("Negative transfer counts found.")

    data["label_differs_from_outcome"] = (
        data["team_label"] != data["final_team_normalized"]
    )
    data["had_transfer"] = data["transfers"] > 0

    print("\n=== OVERALL ROUTING AND TRANSFERS ===")
    print(f"Requests: {len(data):,}")
    print(
        "Initial label differs from final team: "
        f"{data['label_differs_from_outcome'].mean():.2%}"
    )
    print(
        "Requests with at least one recorded transfer: "
        f"{data['had_transfer'].mean():.2%}"
    )
    print(f"Total recorded transfers: {int(data['transfers'].sum()):,}")
    print(f"Average transfers per request: {data['transfers'].mean():.3f}")
    print(f"Median transfers per request: {data['transfers'].median():.1f}")

    print("\n=== TRANSFER BURDEN BY LABEL AGREEMENT ===")
    agreement_table = (
        data.groupby("label_differs_from_outcome")
        .agg(
            requests=("request_id", "size"),
            requests_with_transfer=("had_transfer", "sum"),
            total_transfers=("transfers", "sum"),
            mean_transfers=("transfers", "mean"),
        )
        .reset_index()
    )

    agreement_table["share_with_transfer"] = (
        agreement_table["requests_with_transfer"]
        / agreement_table["requests"]
    )

    agreement_table = agreement_table.rename(
        columns={
            "label_differs_from_outcome": "initial_label_differs_from_final",
        }
    )

    agreement_table.to_csv(
        REPORTS_DIR / "transfer_burden_by_label_agreement.csv",
        index=False,
    )
    print(agreement_table.round(3).to_string(index=False))

    print("\n=== MOST COMMON INITIAL-TO-FINAL TEAM TRANSITIONS ===")
    transitions = (
        data.groupby(
            ["first_team_normalized", "final_team_normalized"],
            dropna=False,
        )
        .agg(
            requests=("request_id", "size"),
            total_transfers=("transfers", "sum"),
            mean_transfers=("transfers", "mean"),
        )
        .reset_index()
        .sort_values("requests", ascending=False)
    )

    transitions.to_csv(
        REPORTS_DIR / "team_transition_counts.csv",
        index=False,
    )
    print(transitions.head(15).round(3).to_string(index=False))

    print("\n=== REQUEST VOLUME AND TRANSFERS BY FINAL TEAM ===")
    workload = (
        data.groupby("final_team_normalized")
        .agg(
            requests_resolved=("request_id", "size"),
            requests_with_transfer=("had_transfer", "sum"),
            total_transfers=("transfers", "sum"),
            mean_transfers=("transfers", "mean"),
        )
        .reset_index()
        .rename(columns={"final_team_normalized": "team"})
        .sort_values("requests_resolved", ascending=False)
    )

    workload["share_of_resolved_requests"] = (
        workload["requests_resolved"] / len(data)
    )
    workload["share_with_transfer"] = (
        workload["requests_with_transfer"]
        / workload["requests_resolved"]
    )

    workload.to_csv(
        REPORTS_DIR / "workload_and_transfers_by_team.csv",
        index=False,
    )
    print(workload.round(3).to_string(index=False))

    print("\nAggregate reports saved in:", REPORTS_DIR)
    print("No customer request text was exported.")


if __name__ == "__main__":
    main()
