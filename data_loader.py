"""
data_loader.py — Handles loading, cleaning, and exporting the RPS data.

This file contains all functions related to getting data in and out of
the program. It can read from CSV and JSON formats, and export to JSON.
"""

import pandas as pd
import json

# Constants
VALID_CHOICES = ["rock", "paper", "scissors"]


def load_csv(filepath):
    """Load a CSV file into a pandas DataFrame and clean it up.

    Steps:
        1. Read the CSV file using pandas.
        2. Strip whitespace from text columns (in case of typos).
        3. Convert choices to lowercase so 'Rock' and 'rock' match.
        4. Remove any rows with missing values.
        5. Remove any rows where the choice isn't rock, paper, or scissors.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The cleaned data, or None if the file wasn't found.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"  Error: Could not find the file '{filepath}'")
        return None

    # Strip whitespace and lowercase the text columns
    df["user"] = df["user"].str.strip()
    df["choice"] = df["choice"].str.strip().str.lower()

    # Count and drop rows with missing data
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  Removed {missing_count} missing values.")
        df = df.dropna()

    # Keep only rows with valid choices (rock, paper, or scissors)
    invalid_mask = ~df["choice"].isin(VALID_CHOICES)
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        print(f"  Removed {invalid_count} invalid choices.")
        df = df[~invalid_mask]

    # Reset the index so row numbers start fresh after removing rows
    df = df.reset_index(drop=True)

    return df


def load_json(filepath):
    """Load data from a JSON file into a pandas DataFrame.

    Expects the JSON to be a list of objects like:
        [{"user": "Dana", "round": 1, "choice": "rock"}, ...]

    Parameters:
        filepath (str): The path to the JSON file.

    Returns:
        pd.DataFrame: The loaded data, or None if file wasn't found.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["choice"] = df["choice"].str.strip().str.lower()
        return df
    except FileNotFoundError:
        print(f"  Error: Could not find the file '{filepath}'")
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
