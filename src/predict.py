from pathlib import Path
import json

import joblib
import pandas as pd


MODEL_FILE = Path("models/energy_forecasting_model.pkl")
CONFIG_FILE = Path("models/feature_config.json")


def load_model_and_config():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Feature configuration not found: {CONFIG_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    return model, config


def predict_energy(
    hour,
    day_of_week,
    month,
    is_weekend,
    lag_1h,
    lag_24h,
    lag_168h,
    rolling_mean_24h,
):
    model, config = load_model_and_config()

    features = pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "lag_1h": lag_1h,
                "lag_24h": lag_24h,
                "lag_168h": lag_168h,
                "rolling_mean_24h": rolling_mean_24h,
            }
        ]
    )

    features = features[config["feature_columns"]]

    prediction = model.predict(features)[0]

    return prediction


def main():
    print("AI Energy Forecasting - Prediction Engine")
    print("=" * 55)

    print("\nLoading saved model...")

    model, config = load_model_and_config()

    print("Saved model loaded successfully.")

    print("\nFeature configuration:")
    for feature in config["feature_columns"]:
        print(f"- {feature}")

    prediction = predict_energy(
        hour=18,
        day_of_week=4,
        month=8,
        is_weekend=0,
        lag_1h=200000,
        lag_24h=205000,
        lag_168h=195000,
        rolling_mean_24h=200000,
    )

    print("\nSample prediction:")
    print(f"Predicted energy demand: {prediction:,.2f} MW")

    print("\n" + "=" * 55)
    print("PREDICTION ENGINE COMPLETED")


if __name__ == "__main__":
    main()