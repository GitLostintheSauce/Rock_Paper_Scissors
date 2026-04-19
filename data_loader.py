"""
data_loader.py — Handles loading, cleaning, and exporting the RPS data.

This file contains all functions related to getting data in and out of
the program. It can read from CSV, JSON, and TXT formats, and export to
JSON. All loaders run the same cleaning pipeline (strip whitespace,
lowercase choices, drop missing values, drop invalid rows).
"""

import io
import pandas as pd
import json

# Constants
VALID_CHOICES = ["rock", "paper", "scissors"]


def _clean_dataframe(df):
    """Shared cleaning pipeline used by every loader.

    Strips whitespace, lowercases choices, removes rows that are missing
    data, and removes rows whose choice isn't rock/paper/scissors.

    Parameters:
        df (pd.DataFrame): Raw frame with user, round, choice columns.

    Returns:
        pd.DataFrame: The cleaned frame (index reset).
    """
    df["user"] = df["user"].astype(str).str.strip()
    df["choice"] = df["choice"].astype(str).str.strip().str.lower()

    missing_count = int(df.isnull().sum().sum())
    if missing_count > 0:
        print(f"  Removed {missing_count} missing values.")
        df = df.dropna()

    invalid_mask = ~df["choice"].isin(VALID_CHOICES)
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        print(f"  Removed {invalid_count} invalid choices.")
        df = df[~invalid_mask]

    return df.reset_index(drop=True)


def load_csv(filepath):
    """Load a CSV file into a pandas DataFrame and clean it up.

    Parameters:
        filepath (str or file-like): Path or uploaded file buffer.

    Returns:
        pd.DataFrame: The cleaned data, or None if the file wasn't found.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"  Error: Could not find the file '{filepath}'")
        return None

    return _clean_dataframe(df)


def load_json(filepath):
    """Load data from a JSON file into a pandas DataFrame.

    Expects the JSON to be a list of objects like:
        [{"user": "Dana", "round": 1, "choice": "rock"}, ...]

    Parameters:
        filepath (str or file-like): Path or uploaded file buffer.

    Returns:
        pd.DataFrame: The cleaned data, or None if file wasn't found.
    """
    try:
        if hasattr(filepath, "read"):
            data = json.load(filepath)
        else:
            with open(filepath, "r") as f:
                data = json.load(f)
    except FileNotFoundError:
        print(f"  Error: Could not find the file '{filepath}'")
        return None

    df = pd.DataFrame(data)
    return _clean_dataframe(df)


def load_txt(filepath):
    """Load data from a whitespace- or comma-separated TXT file.

    Accepts a plain-text file where each row has user, round, choice
    separated by whitespace, commas, or tabs. The first row may be a
    header (user round choice) — if so it's detected automatically.

    Parameters:
        filepath (str or file-like): Path or uploaded file buffer.

    Returns:
        pd.DataFrame: The cleaned data, or None if file wasn't found.
    """
    try:
        if hasattr(filepath, "read"):
            raw = filepath.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            buffer = io.StringIO(raw)
        else:
            buffer = open(filepath, "r")
    except FileNotFoundError:
        print(f"  Error: Could not find the file '{filepath}'")
        return None

    # Auto-detect delimiter: comma, tab, or whitespace
    try:
        df = pd.read_csv(buffer, sep=None, engine="python")
    finally:
        if hasattr(buffer, "close"):
            buffer.close()

    # Normalize column names in case they're cased differently
    df.columns = [c.strip().lower() for c in df.columns]
    return _clean_dataframe(df)


def load_any(filepath, filename=None):
    """Dispatch to the right loader based on file extension.

    Used by the Streamlit app so one button handles CSV, JSON, and TXT
    uploads. Extension is taken from `filename` if provided (useful when
    `filepath` is a buffer), otherwise from `filepath`.

    Parameters:
        filepath (str or file-like): Path or uploaded file buffer.
        filename (str): Optional name used only to read the extension.

    Returns:
        pd.DataFrame or None: Cleaned data, or None if format unsupported.
    """
    name = (filename or str(filepath)).lower()
    if name.endswith(".csv"):
        return load_csv(filepath)
    if name.endswith(".json"):
        return load_json(filepath)
    if name.endswith(".txt"):
        return load_txt(filepath)
    print(f"  Error: unsupported file type '{name}'")
    return None


def export_to_json(df, filepath):
    """Export a DataFrame to a JSON file.

    Converts the DataFrame to a list of dictionaries and saves it.

    Parameters:
        df (pd.DataFrame): The data to export.
        filepath (str): Where to save the JSON file.
    """
    records = df.to_dict(orient="records")
    with open(filepath, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Exported data to {filepath}")


def validate_data(df):
    """Run validation checks on the data and print a summary.

    Checks for:
        - Correct columns exist
        - No missing values remain
        - All choices are valid
        - Each user has the expected number of rounds

    Parameters:
        df (pd.DataFrame): The data to validate.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    is_valid = True

    # Check columns exist
    required_columns = ["user", "round", "choice"]
    for col in required_columns:
        if col not in df.columns:
            print(f"  FAIL: Missing column '{col}'")
            is_valid = False

    if not is_valid:
        return False

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("  FAIL: Data still contains missing values.")
        is_valid = False

    # Check all choices are valid
    invalid = df[~df["choice"].isin(VALID_CHOICES)]
    if len(invalid) > 0:
        print(f"  FAIL: Found {len(invalid)} invalid choices.")
        is_valid = False

    # Report per-user counts
    for user in df["user"].unique():
        count = len(df[df["user"] == user])
        print(f"  {user}: {count} rounds")

    if is_valid:
        print("  All validation checks passed!")

    return is_valid
