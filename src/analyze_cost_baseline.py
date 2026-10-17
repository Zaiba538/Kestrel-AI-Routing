
"""Validate initial-team labels and estimate a historical cost baseline."""

from pathlib import Path

import pandas as pd

from src.data import (
    load_datasets,
    load_reference_files,
    normalize_team_label,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

TRANSFER_COST = 305
EXTRA_CONTACT_COST = 260
ANNUAL_LICENCE_COST = 320_000


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, _ = load_datasets()
    _, resolution = load_reference_files()

    train["request_id"] = train["request_id"].astype(str)
    resolution["request_id"] = resolution["request_id"].astype(str)

    resolution["first_team_normalized"] = resolution["first_team"].map(
        normalize_team_label
    )
    resolution["final_team_normalized"] = resolution["final_team"].map(
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
        raise ValueError("Missing transfer counts.")
    if (data["transfers"] < 0).any():
        raise ValueError("Negative transfer counts found.")

    data["label_matches_first_team"] = (
        data["team_label"] == data["first_team_normalized"]
    )
    data["label_differs_from_final"] = (
        data["team_label"] != data["final_team_normalized"]
    )

    print("\n=== INITIAL-LABEL CONSISTENCY CHECK ===")
    consistent = int(data["label_matches_first_team"].sum())
    inconsistent = int((~data["label_matches_first_team"]).sum())

    print(f"Training requests: {len(data):,}")
    print(f"team_label matches normalized first_team: {consistent:,}")
    print(f"team_label differs from normalized first_team: {inconsistent:,}")

    if inconsistent:
        print("\nMost common inconsistencies:")
        print(
            data.loc[
                ~data["label_matches_first_team"],
                ["team_label", "first_team_normalized"],
            ]
            .value_counts()
            .head(10)
            .to_string()
        )

    requests = len(data)
    months = (
        data["created_at_ist"].max() - data["created_at_ist"].min()
    ).days / 30.44

    if months <= 0:
        raise ValueError("Invalid date range for annualizing costs.")

    annualized_requests = requests / months * 12
    total_transfers = int(data["transfers"].sum())
    differing = int(data["label_differs_from_final"].sum())

    observed_transfer_cost = total_transfers * TRANSFER_COST

    # This is a scenario estimate, not a directly observed contact bill.
    estimated_contact_cost = differing * EXTRA_CONTACT_COST
    scenario_total = observed_transfer_cost + estimated_contact_cost

    annualized_transfer_cost = (
        observed_transfer_cost / requests * annualized_requests
    )
    annualized_scenario_cost = (
        scenario_total / requests * annualized_requests
    )

    print("\n=== HISTORICAL COST BASELINE ===")
    print(f"History: {months:.1f} months")
    print(f"Requests: {requests:,}")
    print(f"Annualized request volume: {annualized_requests:,.0f}")
    print(f"Recorded transfers: {total_transfers:,}")
    print(f"Label/final-team disagreements: {differing:,}")

    print(
        f"\nRecorded transfer cost at Rs {TRANSFER_COST} each: "
        f"Rs {observed_transfer_cost:,.0f}"
    )
    print(
        f"Annualized transfer-cost estimate: "
        f"Rs {annualized_transfer_cost:,.0f}"
    )
    print(
        f"\nExtra-contact scenario at Rs {EXTRA_CONTACT_COST} per "
        f"disagreement: Rs {estimated_contact_cost:,.0f}"
    )
    print(
        f"Combined historical scenario: Rs {scenario_total:,.0f}"
    )
    print(
        f"Annualized combined scenario: Rs {annualized_scenario_cost:,.0f}"
    )
    print(f"Annual licence cost: Rs {ANNUAL_LICENCE_COST:,.0f}")

    print(
        "\nImportant: the extra-contact amount is an assumption applied "
        "to every disagreement, not a measured invoice. The combined "
        "scenario is not proven avoidable savings and excludes the new "
        "system's hosting, maintenance and human-review costs."
    )

    summary = pd.DataFrame([{
        "history_months": months,
        "requests": requests,
        "annualized_requests": annualized_requests,
        "label_first_team_consistency_rate": data[
            "label_matches_first_team"
        ].mean(),
        "recorded_transfers": total_transfers,
        "label_final_disagreements": differing,
        "observed_transfer_cost_rs": observed_transfer_cost,
        "annualized_transfer_cost_rs": annualized_transfer_cost,
        "estimated_extra_contact_cost_rs": estimated_contact_cost,
        "combined_scenario_cost_rs": scenario_total,
        "annualized_combined_scenario_rs": annualized_scenario_cost,
        "annual_licence_cost_rs": ANNUAL_LICENCE_COST,
    }])

    output = REPORTS_DIR / "cost_baseline.csv"
    summary.to_csv(output, index=False)
    print(f"\nSaved aggregate baseline: {output}")


if __name__ == "__main__":
    main()
