from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_FILE = Path("data/energy_features.csv")
MODEL_FILE = Path("models/energy_forecasting_model.pkl")
CONFIG_FILE = Path("models/feature_config.json")
PREDICTIONS_FILE = Path("data/final_predictions.csv")

TRAIN_RATIO = 0.80


def main():
    print("AI Energy Forecasting - Final Model Evaluation")
    print("=" * 55)

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        return

    if not MODEL_FILE.exists():
        print(f"ERROR: Model not found: {MODEL_FILE}")
        return

    if not CONFIG_FILE.exists():
        print(f"ERROR: Feature configuration not found: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    target_column = config["target_column"]
    feature_columns = config["feature_columns"]

    data = pd.read_csv(DATA_FILE)

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(data) * TRAIN_RATIO)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    print(f"Total observations: {len(data):,}")
    print(f"Training observations: {len(train_data):,}")
    print(f"Testing observations: {len(test_data):,}")

    print("\nLoading saved Random Forest model...")

    model = joblib.load(MODEL_FILE)

    print("Saved model loaded successfully.")

    print("\nGenerating final predictions...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    print("Predictions generated.")

    print("\nFinal model performance:")
    print(f"MAE:  {mae:,.2f} MW")
    print(f"RMSE: {rmse:,.2f} MW")
    print(f"R²:   {r2:.4f}")

    comparison = pd.DataFrame(
        {
            "timestamp": test_data["timestamp"].values,
            "actual_mw": y_test.values,
            "predicted_mw": predictions,
        }
    )

    comparison["error_mw"] = (
        comparison["actual_mw"]
        - comparison["predicted_mw"]
    )

    comparison.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print("\nSample final predictions:")

    print(
        comparison.head(5).to_string(index=False)
    )

    print("\nFinal evaluation period:")
    print(f"First: {test_data['timestamp'].min()}")
    print(f"Last:  {test_data['timestamp'].max()}")

    print(f"\nFinal predictions saved: {PREDICTIONS_FILE}")

    print("\n" + "=" * 55)
    print("FINAL MODEL EVALUATION COMPLETED")


if __name__ == "__main__":
    main()