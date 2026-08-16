from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


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
    print("Random Forest Feature Importance")
    print("=" * 50)

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        return

    data = pd.read_csv(DATA_FILE)

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(data) * TRAIN_RATIO)

    train_data = data.iloc[:split_index].copy()

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    print(f"Training observations: {len(train_data):,}")

    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    print("\nFeature importance ranking:")
    print("-" * 50)

    for index, row in importance.iterrows():
        print(
            f"{index + 1}. "
            f"{row['feature']:<20} "
            f"{row['importance']:.4f}"
        )

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()