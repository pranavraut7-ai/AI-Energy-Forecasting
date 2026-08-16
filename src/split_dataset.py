from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/energy_features.csv")

TARGET_COLUMN = "demand_mw"

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
]

TRAIN_RATIO = 0.80


def main():
    print("Energy Forecasting Train/Test Split")
    print("=" * 50)

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        return

    data = pd.read_csv(DATA_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    data = data.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(data) * TRAIN_RATIO)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    print(f"Total observations: {len(data):,}")
    print(f"Training observations: {len(train_data):,}")
    print(f"Testing observations: {len(test_data):,}")

    print("\nTrain period:")
    print(f"First: {train_data['timestamp'].min()}")
    print(f"Last:  {train_data['timestamp'].max()}")

    print("\nTest period:")
    print(f"First: {test_data['timestamp'].min()}")
    print(f"Last:  {test_data['timestamp'].max()}")

    print("\nTarget:")
    print(TARGET_COLUMN)

    print("\nFeatures:")
    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")

    print("\nChronological validation:")

    if train_data["timestamp"].max() < test_data["timestamp"].min():
        print("PASSED: Training data occurs before test data.")
    else:
        print("FAILED: Train/test time overlap detected.")

    if train_data["timestamp"].is_monotonic_increasing:
        print("PASSED: Training data is chronologically ordered.")
    else:
        print("FAILED: Training data is not ordered.")

    if test_data["timestamp"].is_monotonic_increasing:
        print("PASSED: Test data is chronologically ordered.")
    else:
        print("FAILED: Test data is not ordered.")

    print("\nTrain/Test split completed.")
    print("=" * 50)


if __name__ == "__main__":
    main()