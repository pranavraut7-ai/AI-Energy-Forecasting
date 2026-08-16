from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


DATA_FILE = Path("data/energy_features.csv")
MODEL_FILE = Path("models/energy_forecasting_model.pkl")

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
    print("AI Energy Forecasting - Model Persistence")
    print("=" * 55)

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

    print("\nFeatures used:")
    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")

    print("\nTraining selected Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_FILE)

    print(f"\nSaved model: {MODEL_FILE}")

    if MODEL_FILE.exists():
        print(f"Model file size: {MODEL_FILE.stat().st_size:,} bytes")
        print("\nMODEL PERSISTENCE COMPLETED")


if __name__ == "__main__":
    main()