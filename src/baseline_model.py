from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_FILE = Path("data/energy_features.csv")

TARGET_COLUMN = "demand_mw"
BASELINE_COLUMN = "lag_1h"


def main():
    print("Energy Forecasting Baseline Model")
    print("=" * 50)

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        return

    data = pd.read_csv(DATA_FILE)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    data = data.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(data) * 0.80)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    y_train = train_data[TARGET_COLUMN]
    y_test = test_data[TARGET_COLUMN]

    baseline_predictions = test_data[BASELINE_COLUMN]

    mae = mean_absolute_error(y_test, baseline_predictions)

    rmse = mean_squared_error(
        y_test,
        baseline_predictions
    ) ** 0.5

    r2 = r2_score(y_test, baseline_predictions)

    print(f"Total observations: {len(data):,}")
    print(f"Training observations: {len(train_data):,}")
    print(f"Testing observations: {len(test_data):,}")

    print("\nBaseline method:")
    print("Previous hour demand (lag_1h)")

    print("\nTest period:")
    print(f"First: {test_data['timestamp'].min()}")
    print(f"Last:  {test_data['timestamp'].max()}")

    print("\nBaseline performance:")
    print(f"MAE:  {mae:,.2f} MW")
    print(f"RMSE: {rmse:,.2f} MW")
    print(f"R²:   {r2:.4f}")

    print("\nSample predictions:")
    comparison = pd.DataFrame(
        {
            "timestamp": test_data["timestamp"].head(5),
            "actual_mw": y_test.head(5).values,
            "predicted_mw": baseline_predictions.head(5).values,
        }
    )

    print(comparison.to_string(index=False))

    print("\n" + "=" * 50)
    print("BASELINE MODEL COMPLETED")


if __name__ == "__main__":
    main()