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


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)

    rmse = mean_squared_error(
        actual,
        predicted
    ) ** 0.5

    r2 = r2_score(actual, predicted)

    return mae, rmse, r2


def main():
    print("AI Energy Forecasting - Tuned Random Forest")
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

    print(f"Training observations: {len(train_data):,}")
    print(f"Testing observations: {len(test_data):,}")

    print("\nTraining tuned Random Forest...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    predictions = model.predict(X_test)

    tuned_mae, tuned_rmse, tuned_r2 = calculate_metrics(
        y_test,
        predictions
    )

    baseline_predictions = test_data["lag_1h"]

    baseline_mae, baseline_rmse, baseline_r2 = calculate_metrics(
        y_test,
        baseline_predictions
    )

    print("\nTuned model performance:")
    print(f"MAE:  {tuned_mae:,.2f} MW")
    print(f"RMSE: {tuned_rmse:,.2f} MW")
    print(f"R²:   {tuned_r2:.4f}")

    print("\nComparison:")
    print("-" * 55)

    print(f"Baseline MAE:       {baseline_mae:,.2f} MW")
    print(f"Original RF MAE:    2,909.94 MW")
    print(f"Tuned RF MAE:      {tuned_mae:,.2f} MW")

    print()

    print(f"Baseline RMSE:      {baseline_rmse:,.2f} MW")
    print(f"Original RF RMSE:   4,130.31 MW")
    print(f"Tuned RF RMSE:     {tuned_rmse:,.2f} MW")

    print()

    print(f"Baseline R²:        {baseline_r2:.4f}")
    print(f"Original RF R²:     0.8786")
    print(f"Tuned RF R²:       {tuned_r2:.4f}")

    mae_improvement = (
        (2_909.94 - tuned_mae) / 2_909.94
    ) * 100

    print("\nImprovement over original Random Forest:")
    print(f"MAE change: {mae_improvement:+.2f}%")

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
    print("TUNED RANDOM FOREST MODEL COMPLETED")


if __name__ == "__main__":
    main()