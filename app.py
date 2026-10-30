import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="UAC Predictive Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.subheader("HHS Unaccompanied Children (UAC) Program")

st.write(
    "Analyze historical care and discharge data, compare forecasting "
    "models, and explore short-term demand forecasts."
)

CSV_FILE = "HHS_Unaccompanied_Alien_Children_Program_cleaned.csv"

CARE = "Children in HHS Care"
DISCHARGE = "Children discharged from HHS Care"

@st.cache_data
def load_data():
    data = pd.read_csv(CSV_FILE)
    data.columns = data.columns.str.strip()

    required = ["Date", CARE, DISCHARGE]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            "Missing required CSV columns: " + ", ".join(missing)
        )

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    for column in data.columns:
        if column != "Date":
            data[column] = pd.to_numeric(
                data[column].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

    data = data.dropna(subset=["Date"])
    data = data.sort_values("Date")
    data = data.drop_duplicates(subset=["Date"], keep="last")
    data = data.reset_index(drop=True)

    return data


try:
    df = load_data()
except Exception as error:
    st.error("Unable to load the dataset.")
    st.code(str(error))
    st.info(
        "Check that the CSV is uploaded to the same GitHub repository "
        "as app.py and that its filename matches CSV_FILE."
    )
    st.stop()

if df.empty:
    st.error("The dataset contains no valid dated records.")
    st.stop()

if df[CARE].notna().sum() < 10 or df[DISCHARGE].notna().sum() < 10:
    st.error("At least 10 valid observations are required for each KPI.")
    st.stop()

# Sidebar controls
st.sidebar.header("Dashboard Controls")

horizon = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=7,
    max_value=30,
    value=30
)

metric = st.sidebar.selectbox(
    "Select KPI",
    [CARE, DISCHARGE]
)

show_table = st.sidebar.checkbox("Show forecast table", value=True)

model_choice = st.sidebar.selectbox(
    "Forecast model",
    ["7-Day Moving Average", "Naive", "Exponential Smoothing"]
)

# KPI summary
care_values = df[CARE].dropna()
discharge_values = df[DISCHARGE].dropna()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Latest HHS Care Load", f"{care_values.iloc[-1]:,.0f}")
k2.metric("Latest Daily Discharges", f"{discharge_values.iloc[-1]:,.0f}")
k3.metric("Total Records", f"{len(df):,}")
k4.metric("Latest Reporting Date", df["Date"].max().strftime("%d %b %Y"))

st.divider()

# Historical trends
st.subheader("Historical Trends")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Date"], df[CARE], label="Children in HHS Care")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Children")
ax.set_title("Historical HHS Care Load")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Date"], df[DISCHARGE], label="Children Discharged")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Children")
ax.set_title("Historical Discharge Demand")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Forecast functions
def naive_forecast(series, steps):
    return np.repeat(float(series.iloc[-1]), steps)


def moving_average_forecast(series, steps, window=7):
    history = list(series.astype(float))
    predictions = []

    for _ in range(steps):
        prediction = float(np.mean(history[-window:]))
        predictions.append(prediction)
        history.append(prediction)

    return np.array(predictions)


def exponential_smoothing_forecast(series, steps, alpha=0.3):
    values = series.astype(float).to_numpy()
    level = values[0]

    for value in values[1:]:
        level = alpha * value + (1 - alpha) * level

    return np.repeat(level, steps)


def predict(series, model, steps):
    if model == "Naive":
        return naive_forecast(series, steps)

    if model == "7-Day Moving Average":
        return moving_average_forecast(series, steps, window=7)

    return exponential_smoothing_forecast(series, steps, alpha=0.3)


def evaluate_models(series):
    series = series.dropna().reset_index(drop=True)

    if len(series) < 10:
        return pd.DataFrame()

    test_size = min(30, max(1, len(series) // 5))
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    results = []

    for name in [
        "Naive",
        "7-Day Moving Average",
        "Exponential Smoothing"
    ]:
        predictions = predict(train, name, test_size)

        mae = mean_absolute_error(test, predictions)
        rmse = np.sqrt(mean_squared_error(test, predictions))

        actual = test.to_numpy(dtype=float)
        denominator = np.abs(actual)
        errors = np.divide(
            np.abs(actual - predictions),
            denominator,
            out=np.full(len(actual), np.nan),
            where=denominator != 0
        )
        mape = np.nanmean(errors) * 100 if np.any(denominator != 0) else np.nan

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2) if np.isfinite(mape) else np.nan
        })

    return pd.DataFrame(results)


# Model comparison
st.subheader("Forecasting Model Comparison")
st.caption(
    "Models are evaluated on a chronological holdout period. "
    "Lower MAE and RMSE generally indicate better performance."
)

comparison = evaluate_models(df[metric])

if not comparison.empty:
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    best_model = comparison.loc[comparison["MAE"].idxmin(), "Model"]
    st.write(f"Lowest holdout MAE: **{best_model}**")
else:
    st.warning("Not enough valid observations to compare models.")

# Future forecast
st.divider()
st.subheader(f"Future Forecast: {metric}")

series_df = df[["Date", metric]].dropna().copy()
series = series_df[metric].reset_index(drop=True)

forecast_values = predict(series, model_choice, horizon)

last_date = series_df["Date"].max()
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=horizon,
    freq="D"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": np.maximum(forecast_values, 0).round(0).astype(int)
})

st.write(f"Selected model: **{model_choice}**")
st.write(f"Forecast horizon: **{horizon} days**")

fig, ax = plt.subplots(figsize=(12, 5))

history_to_plot = series_df.tail(90)

ax.plot(
    history_to_plot["Date"],
    history_to_plot[metric],
    label="Historical values"
)

ax.plot(
    forecast_df["Date"],
    forecast_df["Forecast"],
    linestyle="--",
    marker="o",
    markersize=3,
    label="Forecast"
)

ax.axvline(last_date, linestyle=":", label="Forecast starts")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Children")
ax.set_title(f"{metric}: Historical Data and Forecast")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

if show_table:
    st.write("Forecast values")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# Download forecast
csv_data = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast CSV",
    data=csv_data,
    file_name="uac_forecast.csv",
    mime="text/csv"
)

st.divider()
st.subheader("Interpretation and Limitations")

st.write(
    "These forecasts are baseline statistical estimates based on historical "
    "values. They do not account for sudden policy changes, border activity, "
    "or other unexpected events. They should support exploratory planning, "
    "not be treated as guaranteed future demand."
)

st.caption(
    "Data source: HHS Unaccompanied Children Program dataset provided "
    "for this project. Forecasts are not official HHS projections."
)
