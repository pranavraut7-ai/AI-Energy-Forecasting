# ⚡ AI Energy Forecasting System

### Smart Grid Energy Demand Prediction & Forecast Analytics

An end-to-end machine learning system for forecasting hourly electricity demand using historical energy-consumption data, time-series feature engineering, and a trained Random Forest regression model.

The project takes historical hourly energy-demand observations, transforms them into predictive time-series features, trains and evaluates a machine learning model using chronological data splitting, persists the selected model, and exposes an interactive Streamlit-based prediction interface.

---

## 📌 Project Overview

Electricity demand changes continuously with time, daily usage patterns, weekly cycles, and historical demand conditions.

Accurately forecasting demand can help energy and grid systems anticipate upcoming load requirements and support better planning and operational decision-making.

This project demonstrates a complete AI forecasting workflow:

```text
Historical Energy Data
        ↓
Data Preparation
        ↓
Feature Engineering
        ↓
Chronological Train/Test Split
        ↓
Baseline Forecast
        ↓
Random Forest Training
        ↓
Feature Importance Analysis
        ↓
Model Evaluation
        ↓
Model Persistence
        ↓
Prediction Engine
        ↓
Interactive Streamlit Application

## 🧠 Machine Learning Approach

The forecasting system uses a supervised machine learning workflow designed for chronological energy-demand data.

### 1. Historical Energy Data

The system starts with historical hourly electricity-demand observations.

The raw data is prepared and validated before being used for model development.

### 2. Feature Engineering

Time-series features are generated to help the model understand temporal demand patterns.

The forecasting feature set includes:

- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `lag_1h`
- `lag_24h`
- `lag_168h`
- `rolling_mean_24h`

The lag features allow the model to use previous demand conditions as signals for future demand.

### 3. Chronological Train/Test Split

Because this is a time-series forecasting problem, the data is split chronologically rather than randomly.

This helps preserve the temporal structure of the dataset and prevents future observations from being used to predict earlier observations.

### 4. Baseline Model

A baseline forecasting model is evaluated before the final machine learning model.

This provides a reference point for assessing whether the trained model provides meaningful predictive performance.

### 5. Random Forest Regression

The final forecasting system uses a **Random Forest Regression** model.

The model learns relationships between engineered time-series features and electricity demand.

The trained model is persisted as:

`models/energy_forecasting_model.pkl`

The feature configuration is stored separately in:

`models/feature_config.json`

---

## 📊 Model Performance

The trained Random Forest forecasting model achieved the following evaluation results:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 2,909.94 MW |
| Root Mean Squared Error (RMSE) | 4,130.31 MW |
| R² Score | 0.8786 |
| Model | Random Forest |

### Metric Interpretation

**MAE — 2,909.94 MW**

Represents the average absolute difference between actual and predicted energy demand.

**RMSE — 4,130.31 MW**

Penalizes larger prediction errors more strongly than MAE.

**R² Score — 0.8786**

Indicates how much of the variation in energy demand is explained by the forecasting model.

---

## 🔍 Feature Importance

Feature importance analysis is performed on the trained Random Forest model to understand which engineered variables contribute most strongly to forecasting decisions.

The analysis shows that:

`lag_1h`

is the dominant predictive feature in the trained model.

This indicates that the most recent energy-demand observation provides a strong signal for forecasting the next demand value.

Other temporal and historical features contribute additional predictive information, including:

- `lag_24h`
- `hour`
- `lag_168h`
- `rolling_mean_24h`
- `day_of_week`
- `month`
- `is_weekend`

---

## 📈 Forecast Analytics

The project includes analytical visualizations for understanding the forecasting system.

The application provides:

- Historical energy-demand trends
- Actual vs predicted demand
- Feature importance analysis
- Model performance metrics
- Forecasting pipeline overview
- Interactive prediction interface

These visualizations provide both a model-development perspective and an operational forecasting perspective.

---

## ⚡ Interactive Forecasting Application

The project includes a Streamlit-based interactive forecasting interface.

The application loads the persisted Random Forest model and allows users to enter current energy-demand conditions.

### Prediction Inputs

The interface accepts:

- Hour
- Day of week
- Month
- Weekend indicator
- Previous hour energy demand
- Previous day same-hour demand
- Previous week same-hour demand
- 24-hour rolling mean demand

After entering the conditions, the forecasting engine generates a predicted energy-demand value in MW.

### Example Prediction

For one tested input configuration, the system generated:

**183,112.61 MW**

The prediction is generated directly through the persisted Random Forest forecasting model.

---

## 🏗️ Project Structure

```text
AI-Energy-Forecasting/
│
├── data/
│   ├── energy_features.csv
│   ├── energy_hourly.csv
│   └── final_predictions.csv
│
├── models/
│   ├── energy_forecasting_model.pkl
│   └── feature_config.json
│
├── src/
│   ├── baseline_model.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── feature_importance.py
│   ├── ml_model.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   ├── save_model.py
│   ├── split_dataset.py
│   ├── tuned_model.py
│   ├── validate_dataset.py
│   └── validate_features.py
│
├── app.py
├── requirements.txt
└── .gitignore