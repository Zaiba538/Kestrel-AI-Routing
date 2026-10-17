
"""Tests for Kestrel assessment data validation."""

from src.data import (
    VALID_TEAMS,
    load_datasets,
    normalize_team_label,
)


def test_training_and_test_data_load():
    train, test = load_datasets()

    assert len(train) == 10822
    assert len(test) == 2178


def test_training_labels_are_valid():
    train, _ = load_datasets()

    assert set(train["team_label"]).issubset(VALID_TEAMS)
    assert train["team_label"].notna().all()


def test_request_ids_are_unique_and_disjoint():
    train, test = load_datasets()

    assert train["request_id"].is_unique
    assert test["request_id"].is_unique
    assert set(train["request_id"].astype(str)).isdisjoint(
        set(test["request_id"].astype(str))
    )


def test_historical_team_names_are_normalized():
    assert normalize_team_label("Installations") == "Installs & Demo"
    assert normalize_team_label("Consumables") == "Filters & Consumables"


def test_test_data_has_no_target_column():
    _, test = load_datasets()

    assert "team_label" not in test.columns
