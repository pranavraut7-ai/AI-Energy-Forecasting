from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_FILE = Path("models/energy_forecasting_model.pkl")
CONFIG_FILE = Path("models/feature_config.json")

HOURLY_DATA_FILE = Path("data/energy_hourly.csv")
PREDICTIONS_FILE = Path("data/final_predictions.csv")


st.set_page_config(
    page_title="AI Energy Forecasting | Smart Grid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
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


@st.cache_data
def load_hourly_data():
    if not HOURLY_DATA_FILE.exists():
        return None

    data = pd.read_csv(HOURLY_DATA_FILE)

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])

    return data


@st.cache_data
def load_predictions():
    if not PREDICTIONS_FILE.exists():
        return None

    data = pd.read_csv(PREDICTIONS_FILE)

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])

    return data


def calculate_model_metrics(predictions):
    if predictions is None:
        return None

    required_columns = {"actual_mw", "predicted_mw"}

    if not required_columns.issubset(predictions.columns):
        return None

    actual = predictions["actual_mw"]
    predicted = predictions["predicted_mw"]

    mae = mean_absolute_error(actual, predicted)

    rmse = mean_squared_error(
        actual,
        predicted,
    ) ** 0.5

    r2 = r2_score(actual, predicted)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def get_feature_importance(model, config):
    feature_columns = config.get(
        "feature_columns",
        [],
    )

    if not hasattr(model, "feature_importances_"):
        return None

    importance_values = model.feature_importances_

    if len(feature_columns) != len(importance_values):
        return None

    importance = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": importance_values,
        }
    )

    importance = importance.sort_values(
        "Importance",
        ascending=False,
    )

    return importance


# -------------------------------------------------------------------
# LOAD SYSTEM
# -------------------------------------------------------------------

try:
    model, config = load_model_and_config()
except Exception as error:
    st.error(
        f"Unable to load forecasting model: {error}"
    )
    st.stop()


hourly_data = load_hourly_data()
predictions_data = load_predictions()

metrics = calculate_model_metrics(
    predictions_data
)

feature_importance = get_feature_importance(
    model,
    config,
)


# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        font-size: 1.15rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .status-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: rgba(20, 180, 100, 0.12);
        border: 1px solid rgba(20, 180, 100, 0.25);
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }

    .section-description {
        opacity: 0.7;
        margin-bottom: 1rem;
    }

    .forecast-number {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
    }

    .forecast-label {
        opacity: 0.7;
        font-size: 0.95rem;
    }

    .small-label {
        font-size: 0.8rem;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="main-title">⚡ AI Energy Forecasting System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Smart Grid Energy Demand Prediction & Forecast Analytics"
    "</div>",
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="status-box">
        🟢 <strong>FORECASTING SYSTEM ONLINE</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Model: <strong>Random Forest</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Forecasting Engine: <strong>Operational</strong>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# SYSTEM OVERVIEW
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">System Overview</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Performance summary of the trained energy-demand forecasting model."
    "</div>",
    unsafe_allow_html=True,
)


kpi1, kpi2, kpi3, kpi4 = st.columns(4)


with kpi1:
    if metrics:
        st.metric(
            "Model R²",
            f"{metrics['r2']:.4f}",
        )
    else:
        st.metric(
            "Model R²",
            "N/A",
        )


with kpi2:
    if metrics:
        st.metric(
            "MAE",
            f"{metrics['mae']:,.0f} MW",
        )
    else:
        st.metric(
            "MAE",
            "N/A",
        )


with kpi3:
    if metrics:
        st.metric(
            "RMSE",
            f"{metrics['rmse']:,.0f} MW",
        )
    else:
        st.metric(
            "RMSE",
            "N/A",
        )


with kpi4:
    if hourly_data is not None:
        st.metric(
            "Hourly Observations",
            f"{len(hourly_data):,}",
        )
    else:
        st.metric(
            "Hourly Observations",
            "N/A",
        )


# -------------------------------------------------------------------
# HISTORICAL ENERGY DEMAND
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Energy Demand Overview</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Historical hourly electricity demand used by the forecasting system."
    "</div>",
    unsafe_allow_html=True,
)


if hourly_data is not None and {
    "timestamp",
    "demand_mw",
}.issubset(hourly_data.columns):

    chart_data = hourly_data[
        ["timestamp", "demand_mw"]
    ].copy()

    chart_data = chart_data.set_index(
        "timestamp"
    )

    st.line_chart(
        chart_data,
        y="demand_mw",
        use_container_width=True,
        height=350,
    )

else:
    st.info(
        "Historical energy dataset is not available."
    )


# -------------------------------------------------------------------
# ACTUAL VS PREDICTED
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Actual vs AI Predicted Demand</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Final test-period comparison between real demand and Random Forest predictions."
    "</div>",
    unsafe_allow_html=True,
)


if predictions_data is not None:

    required_columns = {
        "timestamp",
        "actual_mw",
        "predicted_mw",
    }

    if required_columns.issubset(
        predictions_data.columns
    ):

        comparison_chart = predictions_data[
            [
                "timestamp",
                "actual_mw",
                "predicted_mw",
            ]
        ].copy()

        comparison_chart = comparison_chart.set_index(
            "timestamp"
        )

        st.line_chart(
            comparison_chart,
            y=[
                "actual_mw",
                "predicted_mw",
            ],
            use_container_width=True,
            height=350,
        )

    else:
        st.info(
            "Prediction comparison data is incomplete."
        )

else:
    st.info(
        "Final prediction dataset is not available."
    )


# -------------------------------------------------------------------
# FORECAST ENGINE
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">🔮 Live Forecast Engine</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Enter the latest grid conditions to generate a demand forecast using the trained model."
    "</div>",
    unsafe_allow_html=True,
)


input_col1, input_col2 = st.columns(2)


with input_col1:

    st.markdown("**Time & Calendar Conditions**")

    hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=18,
    )

    day_of_week = st.slider(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=4,
    )

    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=8,
    )

    is_weekend = st.selectbox(
        "Weekend?",
        options=[0, 1],
        format_func=lambda value:
            "Yes" if value == 1 else "No",
    )


with input_col2:

    st.markdown("**Recent Energy Conditions**")

    lag_1h = st.number_input(
        "Previous Hour Energy (MW)",
        min_value=0.0,
        value=200000.0,
        step=1000.0,
    )

    lag_24h = st.number_input(
        "Previous Day Same Hour (MW)",
        min_value=0.0,
        value=205000.0,
        step=1000.0,
    )

    lag_168h = st.number_input(
        "Previous Week Same Hour (MW)",
        min_value=0.0,
        value=195000.0,
        step=1000.0,
    )

    rolling_mean_24h = st.number_input(
        "24-Hour Rolling Mean (MW)",
        min_value=0.0,
        value=200000.0,
        step=1000.0,
    )


predict_clicked = st.button(
    "⚡ Generate Energy Forecast",
    use_container_width=True,
    type="primary",
)


if predict_clicked:

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

    features = features[
        config["feature_columns"]
    ]

    prediction = model.predict(
        features
    )[0]

    st.markdown(
        '<div class="section-title">Forecast Result</div>',
        unsafe_allow_html=True,
    )

    result_col1, result_col2, result_col3 = st.columns(
        3
    )

    with result_col1:

        st.markdown(
            '<div class="small-label">Predicted Energy Demand</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="forecast-number">'
            f"{prediction:,.2f} MW"
            "</div>",
            unsafe_allow_html=True,
        )

    with result_col2:

        st.markdown(
            '<div class="small-label">Forecast Model</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="forecast-number">'
            "Random Forest"
            "</div>",
            unsafe_allow_html=True,
        )

    with result_col3:

        st.markdown(
            '<div class="small-label">Input Demand Baseline</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="forecast-number">'
            f"{rolling_mean_24h:,.0f} MW"
            "</div>",
            unsafe_allow_html=True,
        )

    st.success(
        "Energy demand forecast generated successfully."
    )


# -------------------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">What Drives the Forecast?</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Feature importance from the trained Random Forest model."
    "</div>",
    unsafe_allow_html=True,
)


if feature_importance is not None:

    importance_chart = feature_importance.set_index(
        "Feature"
    )

    st.bar_chart(
        importance_chart,
        y="Importance",
        use_container_width=True,
        height=350,
    )

else:

    st.info(
        "Feature importance is not available for the loaded model."
    )


# -------------------------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Model Performance</div>',
    unsafe_allow_html=True,
)

performance_col1, performance_col2 = st.columns(2)


with performance_col1:

    if metrics:

        st.metric(
            "Mean Absolute Error",
            f"{metrics['mae']:,.2f} MW",
        )

        st.caption(
            "Average absolute difference between actual and predicted demand."
        )

        st.metric(
            "Root Mean Squared Error",
            f"{metrics['rmse']:,.2f} MW",
        )

        st.caption(
            "Penalizes larger prediction errors more strongly."
        )


with performance_col2:

    if metrics:

        st.metric(
            "R² Score",
            f"{metrics['r2']:.4f}",
        )

        st.caption(
            "Measures how much of the demand variation is explained by the model."
        )

    st.metric(
        "Forecasting Model",
        "Random Forest",
    )

    st.caption(
        "Trained on chronological energy-demand data with engineered time-series features."
    )


# -------------------------------------------------------------------
# PROJECT PIPELINE
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Forecasting Pipeline</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "From historical grid data to an AI-generated demand forecast."
    "</div>",
    unsafe_allow_html=True,
)


pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4, pipeline_col5 = st.columns(
    5
)

with pipeline_col1:
    st.markdown("### 01")
    st.write("Historical Data")
    st.caption("Hourly energy demand")

with pipeline_col2:
    st.markdown("### 02")
    st.write("Feature Engineering")
    st.caption("Time & lag features")

with pipeline_col3:
    st.markdown("### 03")
    st.write("Random Forest")
    st.caption("Model training")

with pipeline_col4:
    st.markdown("### 04")
    st.write("Model Evaluation")
    st.caption("MAE • RMSE • R²")

with pipeline_col5:
    st.markdown("### 05")
    st.write("AI Forecast")
    st.caption("Demand prediction")


# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------

st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.caption(
        "AI Energy Forecasting System • Smart Grid Demand Prediction"
    )

with footer_col2:
    st.caption(
        "Machine Learning • Time-Series Features • Random Forest"
    )