from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/energy_features.csv")


def main():
    print("Energy Feature Dataset Validation")
    print("=" * 50)

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        return

    data = pd.read_csv(DATA_FILE)

    print(f"Dataset file: {DATA_FILE}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {list(data.columns)}")

    expected_columns = [
        "timestamp",
        "demand_mw",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_24h",
    ]

    print("\nColumn validation:")

    missing_columns = [
        column for column in expected_columns
        if column not in data.columns
    ]

    unexpected_columns = [
        column for column in data.columns
        if column not in expected_columns
    ]

    if not missing_columns and not unexpected_columns:
        print("PASSED")
    else:
        print("FAILED")

        if missing_columns:
            print(f"Missing columns: {missing_columns}")

        if unexpected_columns:
            print(f"Unexpected columns: {unexpected_columns}")

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    print("\nMissing values:")
    print(f"Timestamp: {data['timestamp'].isna().sum()}")
    print(f"Demand: {data['demand_mw'].isna().sum()}")
    print(f"Lag 1h: {data['lag_1h'].isna().sum()}")
    print(f"Lag 24h: {data['lag_24h'].isna().sum()}")
    print(f"Lag 168h: {data['lag_168h'].isna().sum()}")
    print(f"Rolling mean 24h: {data['rolling_mean_24h'].isna().sum()}")

    print("\nDuplicate timestamps:")
    print(data["timestamp"].duplicated().sum())

    print("\nTime ordering:")
    print(f"Correctly ordered: {data['timestamp'].is_monotonic_increasing}")

    print("\nHourly interval validation:")

    intervals = data["timestamp"].diff().dropna()
    incorrect_intervals = (intervals != pd.Timedelta(hours=1)).sum()

    print(f"Incorrect hourly intervals: {incorrect_intervals}")

    print("\nFeature statistics:")
    print(data.describe().round(3))

    print("\nFirst 5 records:")
    print(data.head().to_string(index=False))

    print("\nLast 5 records:")
    print(data.tail().to_string(index=False))

    print("\n" + "=" * 50)
    print("FEATURE DATASET VALIDATION COMPLETED")


if __name__ == "__main__":
    main()