
"""Data loading and validation for Kestrel Home service routing."""

from pathlib import Path
from typing import Tuple

import pandas as pd
import re


# Project root is the parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test_unlabelled.csv"
TEAMS_FILE = DATA_DIR / "teams.csv"
RESOLUTION_FILE = DATA_DIR / "resolution_log.csv"

TRAIN_COLUMNS = {
    "request_id",
    "created_at_ist",
    "channel",
    "product_family",
    "warranty_status",
    "request_text",
    "source",
    "team_label",
}

TEST_COLUMNS = TRAIN_COLUMNS - {"team_label"}

# Normalize historical team names to their current functional names.
TEAM_ALIASES = {
    "Installations": "Installs & Demo",
    "Installs & Demo": "Installs & Demo",
    "Consumables": "Filters & Consumables",
    "Filters & Consumables": "Filters & Consumables",
}

VALID_TEAMS = {
    "Repairs",
    "Billing",
    "Product Advice",
    "Returns & Replacement",
    "Warranty Claims",
    "Installs & Demo",
    "Filters & Consumables",
}

TEXT_COLUMNS = [
    "channel",
    "product_family",
    "warranty_status",
    "request_text",
    "source",
]


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and raise a clear error if it is unavailable."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Check that the CSV is inside the project's data/ folder."
        )

    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        # Legacy exports may contain non-UTF-8 characters.
        return pd.read_csv(path, encoding="cp1252")


def validate_columns(
    df: pd.DataFrame,
    required_columns: set,
    file_name: str,
) -> None:
    """Check required columns without silently changing the schema."""
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{file_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def normalize_team_label(label: object) -> str:
    """Map historical team names to current functional team names."""
    if pd.isna(label):
        raise ValueError("A team label is missing.")

    label = str(label).strip()

    if label in TEAM_ALIASES:
        return TEAM_ALIASES[label]

    if label in VALID_TEAMS:
        return label

    raise ValueError(f"Unknown team label encountered: {label!r}")

# Mask order and registration identifiers before text enters the model.
# Examples: KO2607918, SR75294, KO-2607918.
ID_PATTERN = re.compile(
    r"\b(?:ko|sr)\s?-?\d{3,}\b",
    flags=re.IGNORECASE,
)


def mask_identifiers(text: str) -> str:
    """Replace order and registration identifiers with a neutral token."""
    return ID_PATTERN.sub("idnum", text)


def normalize_request_text(value: object) -> str:
    """Handle missing text, mask identifiers, and normalize whitespace."""
    if pd.isna(value):
        return ""

    text = mask_identifiers(str(value))
    return " ".join(text.split())



def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare model inputs while preserving the original fields."""
    features = df.copy()

    for column in TEXT_COLUMNS:
        if column in features.columns:
            features[column] = features[column].fillna("").astype(str)

    if "request_text" in features.columns:
        features["request_text"] = features["request_text"].map(
            normalize_request_text
        )

    # Parse creation times for chronological validation.
    # The exported values are displayed in IST.
    if "created_at_ist" in features.columns:
        features["created_at_ist"] = pd.to_datetime(
            features["created_at_ist"],
            errors="coerce",
        )

    return features


def validate_request_ids(df: pd.DataFrame, file_name: str) -> None:
    """Ensure every request has a unique, non-empty identifier."""
    ids = df["request_id"]

    if ids.isna().any():
        raise ValueError(f"{file_name} contains missing request IDs.")

    if ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{file_name} contains empty request IDs.")

    if ids.duplicated().any():
        duplicates = ids[ids.duplicated()].astype(str).head(5).tolist()
        raise ValueError(
            f"{file_name} contains duplicate request IDs. "
            f"Examples: {duplicates}"
        )


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate and prepare labelled and unlabelled datasets."""
    train = load_csv(TRAIN_FILE)
    test = load_csv(TEST_FILE)

    validate_columns(train, TRAIN_COLUMNS, TRAIN_FILE.name)
    validate_columns(test, TEST_COLUMNS, TEST_FILE.name)

    validate_request_ids(train, TRAIN_FILE.name)
    validate_request_ids(test, TEST_FILE.name)

    if train.empty:
        raise ValueError("Training dataset is empty.")

    if test.empty:
        raise ValueError("Test dataset is empty.")

    # Ensure train and test represent different requests.
    overlap = set(train["request_id"].astype(str)) & set(
        test["request_id"].astype(str)
    )

    if overlap:
        raise ValueError(
            f"Train and test contain {len(overlap)} overlapping request IDs."
        )

    train["team_label"] = train["team_label"].map(normalize_team_label)

    unknown_teams = set(train["team_label"]) - VALID_TEAMS

    if unknown_teams:
        raise ValueError(
            f"Unexpected normalized team labels: {sorted(unknown_teams)}"
        )

    train = prepare_features(train)
    test = prepare_features(test)

    return train, test


def load_reference_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load team definitions and the resolution log for analysis."""
    teams = load_csv(TEAMS_FILE)
    resolution = load_csv(RESOLUTION_FILE)

    validate_columns(
        teams,
        {"team", "renamed_to", "handles"},
        TEAMS_FILE.name,
    )

    validate_columns(
        resolution,
        {
            "request_id",
            "first_team",
            "final_team",
            "transfers",
            "resolved_at",
        },
        RESOLUTION_FILE.name,
    )

    if resolution["request_id"].duplicated().any():
        raise ValueError("Resolution log has duplicate request IDs.")

    return teams, resolution


def print_data_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Print a concise dataset health report."""
    print("\n" + "=" * 60)
    print("KESTREL HOME — DATA QUALITY REPORT")
    print("=" * 60)

    print(f"\nTraining rows: {len(train):,}")
    print(f"Test rows:     {len(test):,}")

    print("\nTraining columns:")
    print(list(train.columns))

    print("\nNormalized training label distribution:")
    print(train["team_label"].value_counts().to_string())

    print("\nMissing values in training:")
    print(train.isna().sum().to_string())

    print("\nMissing values in test:")
    print(test.isna().sum().to_string())

    print("\nInvalid creation timestamps:")
    print(f"Train: {train['created_at_ist'].isna().sum():,}")
    print(f"Test:  {test['created_at_ist'].isna().sum():,}")

    print("\nSource distribution:")
    print(train["source"].value_counts(dropna=False).to_string())

    print("\nRequests with empty text:")
    print(f"Train: {(train['request_text'].str.strip() == '').sum():,}")
    print(f"Test:  {(test['request_text'].str.strip() == '').sum():,}")

    print("\nData loading and basic validation completed.")
    print("=" * 60)


if __name__ == "__main__":
    training_data, test_data = load_datasets()
    print_data_report(training_data, test_data)

    print("\nLoading reference files...")
    team_definitions, resolution_log = load_reference_files()

    print(f"Team definition rows: {len(team_definitions)}")
    print(f"Resolution log rows:  {len(resolution_log):,}")
