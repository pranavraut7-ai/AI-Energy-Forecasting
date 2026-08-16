from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/energy_hourly.csv")
OUTPUT_FILE = Path("data/energy_features.csv")


def create_features(data):
    """Create time-based, lag, and rolling features."""

    data = data.copy()

    data["hour"] = data["timestamp"].dt.hour

    data["day_of_week"] = data["timestamp"].dt.dayofweek

    data["month"] = data["timestamp"].dt.month

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["lag_1h"] = data["demand_mw"].shift(1)

    data["lag_24h"] = data["demand_mw"].shift(24)

    data["lag_168h"] = data["demand_mw"].shift(168)

    data["rolling_mean_24h"] = (
        data["demand_mw"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    return data


def main():
    """Create the machine-learning feature dataset."""

    print("Energy Feature Engineering")
    print("=" * 50)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data = data.sort_values("timestamp").reset_index(
        drop=True
    )

    print(f"Input rows: {len(data):,}")

    data = create_features(data)

    feature_columns = [
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

    data = data[feature_columns]

    before_drop = len(data)

    data = data.dropna().reset_index(drop=True)

    removed_rows = before_drop - len(data)

    data["demand_mw"] = data["demand_mw"].round(3)

    data["lag_1h"] = data["lag_1h"].round(3)

    data["lag_24h"] = data["lag_24h"].round(3)

    data["lag_168h"] = data["lag_168h"].round(3)

    data["rolling_mean_24h"] = data[
        "rolling_mean_24h"
    ].round(3)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Rows removed for lag/rolling setup: {removed_rows:,}")
    print(f"Output rows: {len(data):,}")

    print("\nFeatures created:")
    for column in data.columns:
        print(f"  - {column}")

    print("\nFeature dataset created successfully.")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()