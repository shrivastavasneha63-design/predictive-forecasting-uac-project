import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="UAC Predictive Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================
st.title("📊 Predictive Forecasting of Care Load & Placement Demand")
st.markdown(
    "### HHS Unaccompanied Children (UAC) Program"
)

st.write(
    "This dashboard analyzes historical UAC care and discharge data "
    "and provides short-term forecasting to support capacity planning "
    "and placement-demand decisions."
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program_cleaned.csv")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert numeric columns
    numeric_columns = [
        "Children apprehended and placed in CBP custody*",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

    df = df.sort_values("Date").reset_index(drop=True)

    return df


try:
    df = load_data()

except Exception as e:
    st.error("Unable to load the dataset.")
    st.info(
        "Make sure the CSV file is in the same GitHub repository "
        "as app.py and that the filename is correct."
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Dashboard Controls")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=30,
    value=30,
    step=1
)

metric_choice = st.sidebar.selectbox(
    "Select KPI",
    [
        "Children in HHS Care",
        "Children discharged from HHS Care"
    ]
)

show_table = st.sidebar.checkbox(
    "Show forecast table",
    value=True
)


# =========================================================
# DATA SUMMARY
# =========================================================
latest_date = df["Date"].max()

care_col = "Children in HHS Care"
discharge_col = "Children discharged from HHS Care"

latest_care = df[care_col].dropna().iloc[-1]
latest_discharge = df[discharge_col].dropna().iloc[-1]

# =========================================================
# KPI CARDS
# =========================================================
st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Latest HHS Care Load",
        f"{latest_care:,.0f}"
    )

with col2:
    st.metric(
        "Latest Daily Discharges",
        f"{latest_discharge:,.0f}"
    )

with col3:
    st.metric(
        "Records",
        f"{len(df):,}"
    )

with col4:
    st.metric(
        "Latest Date",
        latest_date.strftime("%d %b %Y")
    )


# =========================================================
# HISTORICAL CARE LOAD
# =========================================================
st.subheader("Historical HHS Care Load")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["Date"],
    df[care_col],
    linewidth=2,
    label="Children in HHS Care"
)

ax.set_xlabel("Date")
ax.set_ylabel("Number of Children")
ax.set_title("Historical HHS Care Load")
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


# =========================================================
# HISTORICAL DISCHARGE
# =========================================================
st.subheader("Historical Discharge Demand")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["Date"],
    df[discharge_col],
    linewidth=2,
    label="Children Discharged"
)

ax.set_xlabel("Date")
ax.set_ylabel("Children Discharged")
ax.set_title("Historical HHS Discharge Demand")
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


# =========================================================
# FORECAST FUNCTIONS
# =========================================================

def naive_forecast(series, horizon):
    """
    Naive forecast:
    Future values equal the most recent observed value.
    """
    last_value = series.iloc[-1]

    return np.repeat(last_value, horizon)


def moving_average_forecast(series, horizon, window=7):
    """
    Rolling 7-day moving average forecast.
    """
    history = list(series.dropna().values)

    forecast = []

    for _ in range(horizon):

        recent = history[-window:]

        prediction = np.mean(recent)

        forecast.append(prediction)

        history.append(prediction)

    return np.array(forecast)


def exponential_smoothing_forecast(series, horizon, alpha=0.3):
    """
    Simple exponential smoothing implemented without
    requiring an additional forecasting library.
    """

    values = series.dropna().values.astype(float)

    level = values[0]

    for value in values[1:]:
        level = alpha * value + (1 - alpha) * level

    return np.repeat(level, horizon)


# =========================================================
# MODEL COMPARISON
# =========================================================

def evaluate_models(series):

    series = series.dropna().reset_index(drop=True)

    # Last 30 observations used as holdout
    test_size = min(30, max(7, len(series) // 10))

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    naive_pred = naive_forecast(train, test_size)

    moving_pred = moving_average_forecast(
        train,
        test_size,
        window=7
    )

    ets_pred = exponential_smoothing_forecast(
        train,
        test_size,
        alpha=0.3
    )

    models = {
        "Naive": naive_pred,
        "7-Day Moving Average": moving_pred,
        "Exponential Smoothing": ets_pred
    }

    results = []

    for name, prediction in models.items():

        mae = mean_absolute_error(
            test,
            prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                prediction
            )
        )

        # Avoid division by zero
        denominator = np.where(
            np.abs(test.values) == 0,
            1,
            np.abs(test.values)
        )

        mape = np.mean(
            np.abs(
                (test.values - prediction) /
                denominator
            )
        ) * 100

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape
        })

    return pd.DataFrame(results)


# =========================================================
# MODEL COMPARISON DISPLAY
# =========================================================
st
