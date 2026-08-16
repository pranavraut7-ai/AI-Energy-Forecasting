from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/energy_hourly.csv")


def main():
    """Validate the processed hourly energy dataset."""

    print("Energy Dataset Validation")
    print("=" * 50)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    print(f"Dataset file: {DATA_FILE}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {list(data.columns)}")

    required_columns = [
        "timestamp",
        "demand_mw",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("\nColumn validation: PASSED")

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data["demand_mw"] = pd.to_numeric(
        data["demand_mw"],
        errors="coerce",
    )

    print("\nData types:")
    print(data.dtypes)

    missing_timestamps = data["timestamp"].isna().sum()
    missing_demand = data["demand_mw"].isna().sum()

    print("\nMissing values:")
    print(f"  Timestamp: {missing_timestamps}")
    print(f"  Demand: {missing_demand}")

    duplicate_timestamps = data["timestamp"].duplicated().sum()

    print(
        f"\nDuplicate timestamps: "
        f"{duplicate_timestamps}"
    )

    data = data.sort_values("timestamp")

    timestamp_differences = (
        data["timestamp"]
        .diff()
        .dropna()
    )

    expected_interval = pd.Timedelta(hours=1)

    incorrect_intervals = (
        timestamp_differences != expected_interval
    ).sum()

    print(
        f"Incorrect hourly intervals: "
        f"{incorrect_intervals}"
    )

    print("\nTime coverage:")
    print(f"  First: {data['timestamp'].min()}")
    print(f"  Last:  {data['timestamp'].max()}")

    print("\nDemand statistics:")
    print(
        f"  Minimum: "
        f"{data['demand_mw'].min():,.2f} MW"
    )
    print(
        f"  Maximum: "
        f"{data['demand_mw'].max():,.2f} MW"
    )
    print(
        f"  Mean: "
        f"{data['demand_mw'].mean():,.2f} MW"
    )
    print(
        f"  Median: "
        f"{data['demand_mw'].median():,.2f} MW"
    )

    print("\nFirst 5 records:")
    print(data.head().to_string(index=False))

    print("\nLast 5 records:")
    print(data.tail().to_string(index=False))

    validation_passed = (
        missing_timestamps == 0
        and missing_demand == 0
        and duplicate_timestamps == 0
        and incorrect_intervals == 0
    )

    print("\n" + "=" * 50)

    if validation_passed:
        print("DATASET VALIDATION: PASSED")
    else:
        print("DATASET VALIDATION: FAILED")
        raise ValueError(
            "Dataset contains validation issues."
        )


if __name__ == "__main__":
    main()