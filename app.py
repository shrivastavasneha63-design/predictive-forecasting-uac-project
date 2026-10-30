import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

st.set_page_config(page_title="UAC Predictive Forecasting Dashboard", page_icon="📊", layout="wide")
st.title("Predictive Forecasting of Care Load & Placement Demand")
st.subheader("HHS Unaccompanied Children (UAC) Program")
st.write("Analyze historical UAC care and discharge data, compare forecasting methods, and explore short-term forecasts for capacity planning.")

BASE_DIR = Path(__file__).resolve().parent

def normalize(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())

def find_column(columns, candidates):
    lookup = {normalize(c): c for c in columns}
    for candidate in candidates:
        key = normalize(candidate)
        if key in lookup:
            return lookup[key]
    for candidate in candidates:
        key = normalize(candidate)
        for normalized, original in lookup.items():
            if key and (key in normalized or normalized in key):
                return original
    return None

def find_csv():
    # Accept either the original filename or the cleaned filename.
    names = [
        "HHS_Unaccompanied_Alien_Children_Program.csv",
        "HHS_Unaccompanied_Alien_Children_Program_cleaned.csv",
    ]
    for name in names:
        path = BASE_DIR / name
        if path.exists():
            return path
    files = list(BASE_DIR.glob("*.csv"))
    if len(files) == 1:
        return files[0]
    matches = [p for p in files if "unaccompanied" in p.name.lower()]
    return matches[0] if len(matches) == 1 else None

@st.cache_data
def load_data(path, modified):
    data = pd.read_csv(path)
    date_col = find_column(data.columns, ["Date", "Reporting Date", "Report Date"])
    if date_col is None:
        raise ValueError("Could not find a Date column. Detected columns: " + ", ".join(map(str, data.columns)))
    rename = {date_col: "Date"}
    aliases = {
        "Children apprehended and placed in CBP custody": [
            "Children apprehended and placed in CBP custody",
            "Children apprehended and placed in CBP custody*"
        ],
        "Children in CBP custody": ["Children in CBP custody"],
        "Children transferred out of CBP custody": ["Children transferred out of CBP custody"],
        "Children in HHS Care": ["Children in HHS Care", "Children in HHS care"],
        "Children discharged from HHS Care": [
            "Children discharged from HHS Care",
            "Children discharged from HHS care"
        ],
    }
    for standard, options in aliases.items():
        actual = find_column(data.columns, options)
        if actual is not None:
            rename[actual] = standard
    data = data.rename(columns=rename)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    numeric_cols = [c for c in aliases if c in data.columns]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if not numeric_cols:
        raise ValueError("No expected numeric data columns were found. Check the CSV headings.")
    data = data.sort_values("Date").groupby("Date", as_index=False).last()
    return data.reset_index(drop=True)

csv_file = find_csv()
if csv_file is None:
    st.error("CSV dataset not found.")
    st.info("Upload HHS_Unaccompanied_Alien_Children_Program.csv into the same GitHub folder as app.py. The app also accepts the filename ending in _cleaned.csv.")
    st.stop()

try:
    df = load_data(str(csv_file), csv_file.stat().st_mtime)
except Exception as error:
    st.error("The dataset could not be loaded. Check the CSV filename and column headings.")
    st.exception(error)
    st.stop()

st.caption(f"Dataset loaded: {csv_file.name}")

CARE = "Children in HHS Care"
DISCHARGE = "Children discharged from HHS Care"
if CARE not in df.columns:
    st.error("The column 'Children in HHS Care' was not found.")
    st.write("Columns detected:", list(df.columns))
    st.stop()

st.sidebar.header("Dashboard Controls")
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)
metrics = [CARE] + ([DISCHARGE] if DISCHARGE in df.columns else [])
metric = st.sidebar.selectbox("Select metric to forecast", metrics)
show_table = st.sidebar.checkbox("Show forecast table", True)

latest_care = df[CARE].dropna().iloc[-1]
latest_discharge = df[DISCHARGE].dropna().iloc[-1] if DISCHARGE in df.columns and df[DISCHARGE].notna().any() else None

st.subheader("Key Performance Indicators")
a, b, c, d = st.columns(4)
a.metric("Latest HHS Care Load", f"{latest_care:,.0f}")
b.metric("Latest Daily Discharges", f"{latest_discharge:,.0f}" if latest_discharge is not None else "N/A")
c.metric("Total records", f"{len(df):,}")
d.metric("Latest reporting date", df["Date"].max().strftime("%d %b %Y"))

st.subheader("Historical HHS Care Load")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["Date"], df[CARE], label="Children in HHS Care", linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Number of children")
ax.grid(alpha=0.25)
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

if DISCHARGE in df.columns:
    st.subheader("Historical Discharge Demand")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"], df[DISCHARGE], label="Children discharged", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of children")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

def naive_forecast(series, steps):
    values = series.dropna().astype(float)
    return np.repeat(values.iloc[-1], steps)

def moving_average_forecast(series, steps, window=7):
    history = list(series.dropna().astype(float).values)
    output = []
    for _ in range(steps):
        prediction = float(np.mean(history[-min(window, len(history)):]))
        output.append(prediction)
        history.append(prediction)
    return np.array(output)

def exponential_smoothing_forecast(series, steps, alpha=0.3):
    values = series.dropna().astype(float).values
    level = float(values[0])
    for value in values[1:]:
        level = alpha * float(value) + (1 - alpha) * level
    return np.repeat(level, steps)

def evaluate(series):
    values = series.dropna().astype(float).reset_index(drop=True)
    if len(values) < 10:
        return pd.DataFrame()
    test_size = min(30, max(3, len(values) // 5))
    train, test = values.iloc[:-test_size], values.iloc[-test_size:]
    models = {
        "Naive": naive_forecast(train, test_size),
        "7-Day Moving Average": moving_average_forecast(train, test_size, 7),
        "Exponential Smoothing": exponential_smoothing_forecast(train, test_size, 0.3),
    }
    rows = []
    actual = test.to_numpy()
    for name, prediction in models.items():
        mae = np.mean(np.abs(actual - prediction))
        rmse = np.sqrt(np.mean((actual - prediction) ** 2))
        nonzero = np.abs(actual) > 0
        mape = np.mean(np.abs((actual[nonzero] - prediction[nonzero]) / actual[nonzero])) * 100 if nonzero.any() else np.nan
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

st.subheader("Forecasting Model Comparison")
series = df[metric].dropna().astype(float).reset_index(drop=True)
if len(series) < 10:
    st.warning("At least 10 valid observations are required for model comparison.")
    st.stop()

results = evaluate(series)
st.write("Models are evaluated on the most recent time-ordered holdout observations. Lower MAE and RMSE indicate smaller errors.")
st.dataframe(results.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "MAPE (%)": "{:,.2f}"}), use_container_width=True, hide_index=True)
best = results.iloc[0]["Model"]
st.success(f"Best model on the holdout period based on MAE: {best}")

model_functions = {
    "Naive": naive_forecast,
    "7-Day Moving Average": moving_average_forecast,
    "Exponential Smoothing": exponential_smoothing_forecast,
}
forecast = model_functions[best](series, horizon)
future_dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted value": np.maximum(forecast, 0).round(0).astype(int)
})

st.subheader(f"{horizon}-Day Forecast: {metric}")
fig, ax = plt.subplots(figsize=(12, 4))
history = df[["Date", metric]].dropna().tail(90)
ax.plot(history["Date"], history[metric], label="Historical values", linewidth=2)
ax.plot(forecast_df["Date"], forecast_df["Predicted value"], label=f"Forecast ({best})", marker="o", linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Number of children")
ax.grid(alpha=0.25)
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

if show_table:
    st.write("Forecast values")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

st.download_button(
    "Download forecast as CSV",
    forecast_df.to_csv(index=False).encode("utf-8"),
    "UAC_forecast_results.csv",
    "text/csv"
)

with st.expander("View dataset preview"):
    st.dataframe(df.tail(20), use_container_width=True)

st.markdown("---")
st.caption("Note: These are exploratory statistical forecasts based on historical patterns, not official HHS projections. Actual care load and placement demand may change due to policy, border activity, transfers, and other external factors.")
