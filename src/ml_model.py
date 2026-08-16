from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    print("AI Energy Forecasting - Random Forest Model")
    print("=" * 55)

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        return

    data = pd.read_csv(DATA_FILE)

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(data) * TRAIN_RATIO)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    print(f"Total observations: {len(data):,}")
    print(f"Training observations: {len(train_data):,}")
    print(f"Testing observations: {len(test_data):,}")

    print("\nFeatures used:")
    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")

    print("\nTraining Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    print("\nModel performance:")
    print(f"MAE:  {mae:,.2f} MW")
    print(f"RMSE: {rmse:,.2f} MW")
    print(f"R²:   {r2:.4f}")

    print("\nBaseline comparison:")

    baseline_predictions = test_data["lag_1h"]

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    baseline_rmse = mean_squared_error(
        y_test,
        baseline_predictions
    ) ** 0.5

    baseline_r2 = r2_score(
        y_test,
        baseline_predictions
    )

    print(f"Baseline MAE:  {baseline_mae:,.2f} MW")
    print(f"Model MAE:     {mae:,.2f} MW")

    print(f"\nBaseline RMSE: {baseline_rmse:,.2f} MW")
    print(f"Model RMSE:    {rmse:,.2f} MW")

    print(f"\nBaseline R²:   {baseline_r2:.4f}")
    print(f"Model R²:      {r2:.4f}")

    print("\nSample predictions:")

    comparison = pd.DataFrame(
        {
            "timestamp": test_data["timestamp"].head(5),
            "actual_mw": y_test.head(5).values,
            "predicted_mw": predictions[:5],
        }
    )

    print(comparison.to_string(index=False))

    print("\n" + "=" * 55)
    print("RANDOM FOREST MODEL COMPLETED")


if __name__ == "__main__":
    main()