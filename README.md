# ⚡ AI Energy Forecasting System

### Smart Grid Energy Demand Prediction & Forecast Analytics

An end-to-end machine learning system for forecasting hourly electricity demand using historical energy-consumption data, time-series feature engineering, chronological model evaluation, and Random Forest regression.

The project transforms historical energy-demand observations into predictive time-series features, trains and evaluates a forecasting model, persists the trained model, and provides an interactive Streamlit-based prediction interface.

---

## 📌 Project Overview

Electricity demand changes continuously with time due to daily usage patterns, weekly cycles, and historical demand conditions.

Accurate demand forecasting can help energy and grid systems anticipate upcoming load requirements and support better planning, operational decision-making, and resource management.

This project demonstrates a complete AI-based forecasting workflow:

**Historical Energy Data**  
↓  
**Data Preparation**  
↓  
**Feature Engineering**  
↓  
**Chronological Train/Test Split**  
↓  
**Baseline Forecast**  
↓  
**Random Forest Training**  
↓  
**Feature Importance Analysis**  
↓  
**Model Evaluation**  
↓  
**Model Persistence**  
↓  
**Prediction Engine**  
↓  
**Interactive Streamlit Application**

---

## 🎯 Project Objectives

- Forecast hourly electricity demand using historical observations.
- Prepare and validate time-series energy data.
- Engineer calendar, lag, and rolling statistical features.
- Preserve chronological order during model training and testing.
- Establish a baseline forecasting reference.
- Train a Random Forest regression model.
- Evaluate forecasting performance using regression metrics.
- Analyze the importance of engineered features.
- Persist the trained forecasting model for reuse.
- Provide an interactive interface for generating energy-demand predictions.

---

## 🧠 Machine Learning Approach

The forecasting system follows a structured time-series machine learning workflow.

### 1. Data Preparation

Historical hourly energy-demand observations are loaded and prepared for machine learning.

The preparation stage includes:

- Loading the energy dataset.
- Validating the available observations.
- Handling the time-series structure.
- Preparing the target variable.
- Organizing the data for feature engineering.

### 2. Feature Engineering

The raw time-series data is transformed into predictive features that capture temporal patterns and previous demand conditions.

The engineered features include:

- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `lag_1h`
- `lag_24h`
- `lag_168h`
- `rolling_mean_24h`

The lag features allow the model to use previous demand conditions as signals for future electricity demand.

The calendar features help capture recurring daily, weekly, and monthly demand patterns.

### 3. Chronological Train/Test Split

Because this is a time-series forecasting problem, the dataset is split chronologically rather than randomly.

This preserves the temporal structure of the data and prevents future observations from being used to train the model before they occur in time.

### 4. Baseline Model

A baseline forecasting model is evaluated before the final machine learning model.

The baseline provides a reference point for determining whether the trained Random Forest model provides meaningful predictive improvement.

### 5. Random Forest Regression

The final forecasting system uses a **Random Forest Regression** model.

The model learns relationships between the engineered time-series features and electricity demand.

The trained model is persisted for later prediction use.

### 6. Feature Importance Analysis

Feature importance analysis is used to understand which engineered variables contribute most strongly to the model's predictions.

This provides additional interpretability and helps identify the temporal signals that are most useful for forecasting.

### 7. Model Evaluation

The trained model is evaluated using regression performance metrics, including:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

## 📊 Model Performance

The trained Random Forest forecasting model achieved the following evaluation results:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 2,909.94 MW |
| Root Mean Squared Error (RMSE) | 4,130.31 MW |

These metrics provide a quantitative measure of the difference between predicted and observed electricity-demand values on the evaluation data.

---

## 🔮 Prediction Workflow

The trained model can be used through the prediction engine to generate electricity-demand forecasts.

The prediction process follows these steps:

1. User provides the required input conditions.
2. The system creates the corresponding time-series features.
3. The persisted Random Forest model receives the engineered features.
4. The model generates the predicted energy-demand value.
5. The prediction is presented in MW.

### Example Prediction

For one tested input configuration, the system generated:

**183,112.61 MW**

The prediction is generated directly through the persisted Random Forest forecasting model.

---

## 🖥️ Interactive Application

The project includes a Streamlit-based application that provides an interactive interface for the forecasting system.

The application connects the trained forecasting model with the prediction engine so that users can provide input conditions and obtain an energy-demand forecast.

The application is launched through:

```bash
streamlit run app.py

## 🗂️ Project Structure

The project is organized into separate components for data, trained models, machine learning workflows, validation, prediction, and application execution.

### Data

- `data/energy_hourly.csv` — historical hourly energy-demand data.
- `data/energy_features.csv` — engineered time-series feature dataset.
- `data/final_predictions.csv` — generated prediction results.

### Models

- `models/energy_forecasting_model.pkl` — persisted Random Forest forecasting model.
- `models/feature_config.json` — stored feature configuration used by the prediction workflow.

### Source Code

- `src/validate_dataset.py` — validates the dataset structure and data quality.
- `src/prepare_dataset.py` — prepares the raw energy data for modelling.
- `src/feature_engineering.py` — creates time-series and calendar-based features.
- `src/validate_features.py` — validates the engineered feature set.
- `src/split_dataset.py` — performs the chronological train/test split.
- `src/baseline_model.py` — evaluates the baseline forecasting approach.
- `src/ml_model.py` — trains the Random Forest regression model.
- `src/tuned_model.py` — handles the tuned model workflow.
- `src/feature_importance.py` — analyzes model feature importance.
- `src/evaluate_model.py` — evaluates forecasting performance.
- `src/save_model.py` — persists the trained forecasting model.
- `src/predict.py` — generates predictions using the persisted model.

### Application

- `app.py` — Streamlit application for interactive energy-demand forecasting.

### Configuration

- `requirements.txt` — Python dependencies required to run the project.
- `.gitignore` — files and directories excluded from Git tracking.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Model | Random Forest Regression |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Application Interface | Streamlit |
| Model Persistence | Joblib |
| Version Control | Git |
| Repository Hosting | GitHub |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/pranavraut7-ai/AI-Energy-Forecasting.git