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

Date,Children apprehended and placed in CBP custody*,Children in CBP custody,Children transferred out of CBP custody,Children in HHS Care,Children discharged from HHS Care
"December 21, 2025",6,18,11,"2,484",14
"December 18, 2025",11,50,6,"2,472",16
"December 17, 2025",7,31,11,"2,481",10
"December 16, 2025",8,54,15,"2,468",9
"December 15, 2025",11,42,9,"2,470",7
"December 14, 2025",8,35,4,"2,462",8
"December 11, 2025",7,47,9,"2,437",10
"December 10, 2025",10,54,5,"2,439",9
"December 09, 2025",4,30,7,"2,443",8
"December 08, 2025",9,27,9,"2,440",4
"December 07, 2025",4,24,13,"2,429",12
"December 04, 2025",16,39,7,"2,418",13
"December 03, 2025",4,32,8,"2,419",7
"December 02, 2025",7,34,6,"2,407",8
"December 01, 2025",4,26,4,"2,410",9
"November 30, 2025",4,18,10,"2,415",0
"November 27, 2025",4,24,8,"2,389",7
"November 25, 2025",6,47,14,"2,375",11
"November 24, 2025",9,40,9,"2,371",13
"November 23, 2025",3,46,22,"2,375",1
"November 20, 2025",18,57,8,"2,340",9
"November 19, 2025",7,32,3,"2,340",10
"November 18, 2025",2,38,8,"2,345",12
"November 17, 2025",1,35,6,"2,343",8
"November 16, 2025",5,22,3,"2,340",5
"November 13, 2025",6,26,16,"2,317",5
"November 12, 2025",14,48,20,"2,302",7
"November 11, 2025",11,39,12,"2,294",14
"November 09, 2025",11,33,5,"2,294",4
"November 06, 2025",6,33,9,"2,264",19
"November 05, 2025",8,41,8,"2,272",7
"November 04, 2025",6,33,6,"2,270",8
"November 03, 2025",3,28,5,"2,269",12
"November 02, 2025",5,21,10,"2,277",7
"October 30, 2025",3,17,14,"2,287",12
"October 29, 2025",6,25,4,"2,280",17
"October 28, 2025",6,22,11,"2,286",6
"October 27, 2025",2,31,8,"2,282",11
"October 26, 2025",4,35,12,"2,276",10
"October 23, 2025",4,27,26,"2,266",18
"October 22, 2025",12,42,13,"2,261",12
"October 21, 2025",13,39,17,"2,251",9
"October 20, 2025",10,39,6,"2,243",7
"October 19, 2025",9,34,17,"2,237",7
"October 16, 2025",9,33,15,"2,233",9
"October 15, 2025",8,29,10,"2,225",15
"October 14, 2025",7,30,16,"2,235",8
"October 13, 2025",8,24,7,"2,223",6
"October 09, 2025",11,55,13,"2,192",12
"October 08, 2025",8,49,15,"2,192",17
"October 07, 2025",11,45,4,"2,189",10
"October 06, 2025",5,33,15,"2,187",10
"October 05, 2025",8,28,13,"2,180",16
"October 02, 2025",20,51,15,"2,164",16
"October 01, 2025",4,33,12,"2,163",14
"September 30, 2025",8,32,12,"2,159",7
"September 29, 2025",8,35,12,"2,148",13
"September 28, 2025",11,29,12,"2,146",9
"September 25, 2025",6,35,12,"2,132",18
"September 24, 2025",3,38,12,"2,138",17
"September 23, 2025",11,36,10,"2,145",15
"September 22, 2025",17,40,14,"2,141",10
"September 21, 2025",8,35,13,"2,134",5
"September 18, 2025",20,52,11,"2,096",14
"September 17, 2025",7,38,5,"2,096",13
"September 16, 2025",7,28,17,"2,095",12
"September 15, 2025",9,47,10,"2,086",10
"September 14, 2025",7,39,13,"2,083",5
"September 11, 2025",12,38,29,"2,047",6
"September 10, 2025",10,47,14,"2,025",18
"September 09, 2025",18,42,11,"2,023",14
"September 08, 2025",7,25,5,"2,025",6
"September 07, 2025",4,19,9,"2,026",7
"September 04, 2025",11,40,2,"2,015",14
"September 03, 2025",4,29,11,"2,019",10
"September 02, 2025",6,35,9,"2,020",9
"September 01, 2025",10,29,17,"2,010",5
"August 28, 2025",8,33,6,"1,984",17
"August 27, 2025",5,19,9,"1,988",19
"August 26, 2025",10,21,20,"1,989",15
"August 25, 2025",8,30,8,"1,980",11
"August 24, 2025",4,21,5,"1,981",5
"August 21, 2025",6,49,8,"1,972",21
"August 20, 2025",10,38,10,"1,983",10
"August 19, 2025",5,35,3,"1,980",11
"August 18, 2025",4,20,6,"1,985",14
"August 17, 2025",4,14,2,"1,991",3
"August 14, 2025",3,27,4,"1,994",11
"August 13, 2025",6,22,12,"1,997",17
"August 12, 2025",8,38,9,"1,999",9
"August 11, 2025",6,23,4,"1,999",16
"August 10, 2025",0,18,8,"2,006",12
"August 07, 2025",4,18,6,"2,031",17
"August 06, 2025",4,17,3,"2,036",13
"August 05, 2025",2,21,10,"2,042",14
"August 04, 2025",5,22,11,"2,040",14
"August 03, 2025",5,26,3,"2,049",5
"July 31, 2025",2,9,6,"2,058",15
"July 30, 2025",4,15,4,"2,063",16
"July 29, 2025",2,8,3,"2,072",12
"July 28, 2025",2,9,3,"2,080",14
"July 27, 2025",1,12,2,"2,088",4
"July 24, 2025",6,18,7,"2,096",29
"July 23, 2025",6,21,5,"2,111",14
"July 22, 2025",5,13,3,"2,117",16
"July 21, 2025",4,18,7,"2,121",16
"July 17, 2025",3,16,2,"2,153",17
"July 16, 2025",1,9,0,"2,164",11
"July 15, 2025",0,8,5,"2,171",17
"July 14, 2025",2,9,1,"2,181",26
"July 13, 2025",1,13,4,"2,204",10
"July 10, 2025",2,12,3,"2,252",39
"July 09, 2025",2,18,0,"2,281",32
"July 08, 2025",5,16,8,"2,306",20
"July 07, 2025",3,18,8,"2,317",20
"July 06, 2025",4,17,2,"2,332",8
"July 02, 2025",8,20,9,"2,373",38
"July 01, 2025",7,14,6,"2,391",22
"June 30, 2025",9,22,8,"2,407",16
"June 29, 2025",5,17,7,"2,413",20
"June 26, 2025",3,9,7,"2,447",35
"June 25, 2025",6,12,2,"2,470",24
"June 24, 2025",5,20,5,"2,489",16
"June 23, 2025",3,15,5,"2,495",17
"June 22, 2025",7,13,5,"2,508",7
"June 19, 2025",8,19,7,"2,501",18
"June 17, 2025",7,24,10,"2,523",14
"June 10, 2025",17,28,7,"2,511",12
"June 09, 2025",8,21,14,"2,516",9
"June 08, 2025",10,16,15,"2,507",13
"June 05, 2025",8,26,5,"2,503",21
"June 04, 2025",4,16,9,"2,516",16
"June 03, 2025",5,23,11,"2,519",15
"June 02, 2025",10,19,5,"2,518",16
"June 01, 2025",8,24,16,"2,528",12
"May 29, 2025",13,38,15,"2,506",11
"May 28, 2025",12,39,12,"2,499",14
"May 27, 2025",6,24,7,"2,504",11
"May 22, 2025",18,36,17,"2,473",15
"May 21, 2025",14,62,13,"2,467",18
"May 19, 2025",7,27,17,"2,457",6
"May 18, 2025",13,28,14,"2,437",9
"May 15, 2025",15,31,14,"2,410",14
"May 14, 2025",7,30,14,"2,394",11
"May 13, 2025",18,29,12,"2,394",11
"May 12, 2025",7,33,14,"2,392",7
"May 11, 2025",15,20,12,"2,385",3
"May 08, 2025",15,24,16,"2,361",11
"May 07, 2025",10,30,15,"2,335",10
"May 06, 2025",10,33,16,"2,339",16
"May 05, 2025",15,44,20,"2,331",17
"May 04, 2025",9,31,10,"2,335",5
"May 01, 2025",13,32,14,"2,306",16
"April 30, 2025",7,34,9,"2,309",12
"April 29, 2025",11,25,11,"2,304",15
"April 28, 2025",6,26,12,"2,307",8
"April 27, 2025",9,40,12,"2,301",4
"April 24, 2025",3,9,7,"2,294",13
"April 23, 2025",3,23,4,"2,300",11
"April 22, 2025",7,19,14,"2,305",17
"April 21, 2025",10,25,5,"2,308",6
"April 20, 2025",4,15,6,"2,305",4
"April 17, 2025",7,26,10,"2,288",14
"April 16, 2025",11,30,5,"2,291",8
"April 15, 2025",4,19,12,"2,286",7
"April 14, 2025",5,26,9,"2,282",13
"April 13, 2025",5,26,10,"2,283",3
"April 11, 2025",8,22,8,"2,267",9
"April 10, 2025",18,25,17,"2,265",6
"April 09, 2025",6,19,10,"2,251",5
"April 08, 2025",12,34,12,"2,243",9
"April 07, 2025",12,22,9,"2,237",5
"April 06, 2025",7,14,7,"2,234",11
"April 03, 2025",12,26,18,"2,223",8
"April 02, 2025",5,31,16,"2,208",3
"April 01, 2025",17,36,7,"2,191",12
"March 31, 2025",9,25,8,"2,193",5
"March 30, 2025",9,20,5,"2,188",2
"March 27, 2025",1,19,13,"2,159",2
"March 26, 2025",9,21,15,"2,159",2
"March 25, 2025",12,19,7,"2,145",6
"March 24, 2025",7,13,11,"2,139",7
"March 23, 2025",15,29,9,"2,136",2
"March 20, 2025",9,20,11,"2,124",7
"March 19, 2025",6,22,14,"2,119",9
"March 18, 2025",19,34,6,"2,109",8
"March 17, 2025",5,14,0,"2,111",8
"March 16, 2025",1,9,8,"2,117",5
"March 13, 2025",4,20,9,"2,116",6
"March 12, 2025",6,18,2,"2,113",8
"March 11, 2025",3,11,7,"2,116",20
"March 10, 2025",6,28,8,"2,125",23
"March 09, 2025",12,25,15,"2,137",30
"March 06, 2025",7,27,6,"2,218",33
"March 05, 2025",5,20,4,"2,238",40
"March 04, 2025",4,25,10,"2,272",30
"March 03, 2025",8,29,18,"2,287",25
"March 02, 2025",12,24,4,"2,297",31
"February 27, 2025",10,15,17,"2,391",37
"February 26, 2025",12,21,1,"2,411",23
"February 25, 2025",11,7,7,"2,413",23
"February 24, 2025",20,22,14,"2,413",24
"February 23, 2025",7,16,8,"2,423",35
"February 20, 2025",13,25,15,"2,520",32
"February 19, 2025",12,17,17,"2,528",19
"February 18, 2025",18,19,11,"2,536",9
"February 17, 2025",8,17,14,"2,532",15
"February 13, 2025",15,10,23,"2,703",72
"February 12, 2025",16,20,20,"2,752",66
"February 11, 2025",11,22,12,"2,799",50
"February 10, 2025",16,18,15,"2,834",63
"February 09, 2025",15,13,23,"2,878",99
"February 06, 2025",25,32,27,"3,181",145
"February 05, 2025",26,36,36,"3,290",104
"February 04, 2025",34,54,29,"3,359",74
"February 03, 2025",23,45,11,"3,403",102
"February 02, 2025",20,22,41,"3,483",168
"January 30, 2025",47,42,47,"3,923",159
"January 29, 2025",32,50,39,"4,036",99
"January 28, 2025",32,49,39,"4,096",101
"January 27, 2025",33,53,34,"4,161",127
"January 26, 2025",29,51,11,"4,241",182
"January 23, 2025",91,75,97,"4,621",174
"January 22, 2025",62,97,52,"4,717",83
"January 21, 2025",59,64,117,"4,728",54
"January 20, 2025",84,118,136,"4,638",127
"January 19, 2025",86,152,168,"4,628",176
"January 16, 2025",106,196,131,"4,881",159
"January 15, 2025",113,162,107,"4,903",164
"January 14, 2025",92,124,93,"4,957",139
"January 13, 2025",68,107,102,"5,019",181
"January 12, 2025",74,103,80,"5,092",235
"January 09, 2025",56,82,68,"5,507",164
"January 08, 2025",38,80,82,"5,605",184
"January 07, 2025",65,83,65,"5,703",133
"January 06, 2025",41,60,71,"5,769",190
"January 05, 2025",43,73,64,"5,882",185
"January 02, 2025",35,70,53,"6,292",170
"January 01, 2025",25,66,94,"6,405",144
"December 30, 2024",99,156,139,"6,454",177
"December 29, 2024",107,170,133,"6,465",216
"December 26, 2024",80,122,152,"6,707",135
"December 25, 2024",57,162,193,"6,689",107
"December 24, 2024",161,304,211,"6,622",155
"December 23, 2024",205,294,177,"6,548",231
"December 22, 2024",100,255,236,"6,595",268
"December 19, 2024",269,412,204,"6,600",238
"December 18, 2024",168,285,210,"6,598",134
"December 17, 2024",187,310,247,"6,506",151
"December 16, 2024",133,287,189,"6,454",149
"December 15, 2024",180,337,219,"6,408",224
"December 12, 2024",153,253,286,"6,481",194
"December 11, 2024",209,353,232,"6,402",152
"December 10, 2024",167,326,213,"6,310",149
"December 09, 2024",224,315,152,"6,268",174
"December 08, 2024",132,243,183,"6,286",172
"December 05, 2024",218,283,208,"6,310",150
"December 04, 2024",118,245,160,"6,269",115
"December 03, 2024",81,230,242,"6,192",107
"December 02, 2024",236,330,222,"6,061",115
"December 01, 2024",196,306,187,"5,986",149
"November 28, 2024",113,280,237,"5,963",143
"November 27, 2024",239,367,224,"5,862",211
"November 26, 2024",152,272,214,"5,826",112
"November 25, 2024",170,297,196,"5,715",129
"November 24, 2024",191,283,192,"5,664",207
"November 21, 2024",109,214,173,"5,850",210
"November 20, 2024",155,212,181,"5,879",148
"November 19, 2024",119,209,152,"5,844",146
"November 18, 2024",143,203,163,"5,819",138
"November 17, 2024",105,210,149,"5,816",195
"November 14, 2024",119,219,210,"6,025",172
"November 13, 2024",148,245,175,"5,966",151
"November 12, 2024",142,235,139,"5,975",138
"November 11, 2024",118,139,168,"5,963",145
"November 07, 2024",125,232,192,"6,084",177
"November 06, 2024",156,256,176,"6,086",137
"November 05, 2024",109,245,159,"6,045",126
"November 04, 2024",167,209,157,"6,007",132
"November 03, 2024",84,171,157,"5,990",184
"October 31, 2024",87,246,173,"6,148",177
"October 30, 2024",136,276,178,"6,149",160
"October 29, 2024",166,277,229,"6,106",126
"October 28, 2024",110,272,193,"6,008",156
"October 27, 2024",202,313,158,"5,974",192
"October 23, 2024",134,294,186,"6,077",138
"October 22, 2024",180,280,146,"6,043",109
"October 21, 2024",108,220,213,"5,966",154
"October 20, 2024",147,276,214,"5,918",195
"October 17, 2024",104,230,206,"6,076",160
"October 16, 2024",174,285,150,"6,036",129
"October 15, 2024",113,208,159,"6,013",107
"October 14, 2024",88,214,160,"5,935",106
"October 13, 2024",130,227,192,"5,889",210
"October 10, 2024",84,158,193,"6,075",159
"October 09, 2024",115,216,169,"6,047",137
"October 08, 2024",167,239,166,"6,013",123
"October 07, 2024",99,190,176,"5,968",153
"October 03, 2024",101,162,178,"6,118",193
"October 02, 2024",107,215,230,"6,091",173
"October 01, 2024",184,314,217,"6,040",137
"September 30, 2024",209,285,184,"5,967",138
"September 29, 2024",81,213,153,"5,925",177
"September 26, 2024",91,199,172,"6,090",185
"September 25, 2024",154,225,220,"6,050",169
"September 24, 2024",186,259,172,"6,023",143
"September 23, 2024",92,203,173,"6,002",165
"September 22, 2024",124,234,161,"5,989",197
"September 19, 2024",93,196,169,"6,270",188
"September 18, 2024",116,213,217,"6,245",149
"September 17, 2024",159,304,139,"6,227",160
"September 16, 2024",113,209,160,"6,229",164
"September 15, 2024",98,220,167,"6,234",210
"September 13, 2024",81,200,212,"6,326",279
"September 12, 2024",184,280,210,"6,360",220
"September 11, 2024",93,248,175,"6,410",162
"September 10, 2024",131,251,180,"6,396",148
"September 09, 2024",144,275,190,"6,353",141
"September 08, 2024",123,255,164,"6,312",221
"September 05, 2024",126,230,175,"6,448",121
"September 04, 2024",121,231,219,"6,375",114
"September 03, 2024",131,263,177,"6,272",112
"September 02, 2024",133,270,198,"6,192",129
"August 29, 2024",145,295,188,"6,250",169
"August 28, 2024",170,285,178,"6,227",134
"August 27, 2024",149,285,222,"6,152",97
"August 22, 2024",182,307,205,"6,054",171
"August 21, 2024",134,245,167,"6,028",156
"August 20, 2024",136,274,245,"5,995",112
"August 19, 2024",176,254,162,"5,865",158
"August 18, 2024",86,244,235,"5,821",178
"August 15, 2024",124,241,156,"5,895",220
"August 14, 2024",109,226,191,"5,987",178
"August 13, 2024",162,239,189,"5,981",123
"August 12, 2024",109,219,155,"5,903",148
"August 11, 2024",95,211,140,"5,894",249
"August 08, 2024",119,230,213,"6,084",174
"August 07, 2024",135,273,207,"6,038",174
"August 06, 2024",163,283,200,"5,998",143
"August 05, 2024",172,259,138,"5,950",150
"August 04, 2024",63,167,143,"5,943",170
"August 01, 2024",227,298,212,"6,016",169
"July 31, 2024",140,261,163,"5,963",154
"July 30, 2024",125,203,208,"5,930",92
"July 29, 2024",114,210,219,"5,819",163
"July 28, 2024",152,290,199,"5,743",207
"July 25, 2024",93,198,208,"5,954",194
"July 24, 2024",137,272,166,"5,927",177
"July 23, 2024",120,257,126,"5,965",170
"July 22, 2024",102,189,155,"6,002",234
"July 21, 2024",90,196,205,"6,058",262
"July 17, 2024",128,228,177,"6,228",187
"July 16, 2024",136,246,123,"6,239",173
"July 15, 2024",105,185,158,"6,283",171
"July 14, 2024",116,188,164,"6,308",237
"July 11, 2024",80,176,125,"6,640",197
"July 10, 2024",150,157,201,"6,649",177
"July 09, 2024",116,181,182,"6,651",130
"July 08, 2024",141,173,129,"6,617",123
"July 07, 2024",96,160,233,"6,599",213
"July 04, 2024",158,226,191,"6,790",225
"July 02, 2024",141,232,188,"6,823",184
"July 01, 2024",123,233,159,"6,825",229
"June 30, 2024",123,230,167,"6,874",273
"June 27, 2024",174,208,256,"7,248",268
"June 26, 2024",132,235,197,"7,268",187
"June 25, 2024",172,263,191,"7,239",202
"June 24, 2024",116,175,172,"7,239",202
"June 23, 2024",102,227,201,"7,247",294
"June 20, 2024",145,247,243,"7,431",235
"June 19, 2024",204,321,246,"7,433",198
"June 18, 2024",160,297,253,"7,340",214
"June 17, 2024",159,310,272,"7,318",224
"June 16, 2024",204,315,233,"7,307",294
"June 13, 2024",167,293,230,"7,544",259
"June 12, 2024",152,266,274,"7,548",212
"June 11, 2024",186,352,224,"7,493",156
"June 10, 2024",184,334,229,"7,439",212
"June 09, 2024",116,289,166,"7,451",312
"June 06, 2024",179,326,282,"7,741",238
"June 05, 2024",173,345,291,"7,691",198
"June 04, 2024",260,356,176,"7,565",203
"June 03, 2024",109,244,257,"7,596",224
"June 02, 2024",159,330,322,"7,546",306
"May 30, 2024",167,406,339,"7,563",190
"May 29, 2024",182,462,158,"7,491",141
"May 28, 2024",153,348,239,"7,418",151
"May 27, 2024",156,344,252,"7,365",167
"May 23, 2024",193,272,314,"7,494",227
"May 22, 2024",168,349,272,"7,474",208
"May 21, 2024",196,348,176,"7,404",145
"May 20, 2024",111,268,217,"7,368",192
"May 19, 2024",143,295,278,"7,356",298
"May 16, 2024",206,340,214,"7,565",234
"May 15, 2024",115,285,341,"7,589",265
"May 14, 2024",246,419,218,"7,546",186
"May 13, 2024",112,331,167,"7,499",157
"May 12, 2024",138,301,263,"7,464",245
"May 09, 2024",138,285,268,"7,570",244
"May 08, 2024",167,374,205,"7,583",202
"May 07, 2024",176,334,246,"7,566",163
"May 06, 2024",173,337,211,"7,449",205
"May 05, 2024",150,315,228,"7,447",332
"May 02, 2024",206,411,293,"7,577",194
"May 01, 2024",197,437,295,"7,508",190
"April 30, 2024",220,443,194,"7,431",135
"April 29, 2024",140,292,255,"7,373",187
"April 28, 2024",158,349,365,"7,298",250
"April 25, 2024",145,357,333,"7,209",217
"April 24, 2024",253,411,307,"7,128",154
"April 23, 2024",252,427,307,"6,967",134
"April 22, 2024",176,424,251,"6,799",214
"April 21, 2024",208,380,343,"6,745",240
"April 18, 2024",144,322,258,"6,728",260
"April 17, 2024",182,371,274,"6,725",227
"April 16, 2024",219,392,205,"6,685",163
"April 15, 2024",144,303,176,"6,681",191
"April 14, 2024",128,245,243,"6,630",290
"April 11, 2024",183,221,220,"6,912",285
"April 10, 2024",117,244,142,"6,999",227
"April 09, 2024",118,231,217,"7,043",130
"April 08, 2024",102,230,167,"6,988",199
"April 07, 2024",106,239,233,"6,982",279
"April 04, 2024",95,226,152,"7,317",225
"April 03, 2024",124,230,190,"7,346",163
"April 02, 2024",101,230,175,"7,340",132
"April 01, 2024",109,235,171,"7,271",172
"March 31, 2024",105,254,151,"7,257",287
"March 28, 2024",161,288,248,"7,762",360
"March 27, 2024",129,287,185,"7,893",215
"March 26, 2024",187,276,200,"7,919",208
"March 25, 2024",128,242,224,"7,899",229
"March 24, 2024",124,264,221,"7,899",373
"March 21, 2024",132,261,141,"8,292",235
"March 20, 2024",91,181,174,"8,365",227
"March 19, 2024",99,202,256,"8,390",197
"March 18, 2024",123,243,188,"8,333",283
"March 17, 2024",127,260,259,"8,373",342
"March 14, 2024",166,301,252,"8,695",310
"March 13, 2024",165,358,167,"8,819",239
"March 12, 2024",154,294,290,"8,843",213
"March 11, 2024",178,336,290,"8,757",253
"March 10, 2024",190,368,326,"8,714",373
"March 07, 2024",181,424,328,"8,902",305
"March 06, 2024",230,457,277,"8,870",232
"March 05, 2024",246,472,345,"8,800",214
"March 04, 2024",225,509,340,"8,660",229
"March 03, 2024",236,468,347,"8,594",354
"February 29, 2024",262,444,354,"8,724",299
"February 28, 2024",209,447,274,"8,699",257
"February 27, 2024",197,369,306,"8,650",206
"February 26, 2024",199,438,326,"8,404",329
"February 25, 2024",244,482,366,"8,404",329
"February 22, 2024",165,400,292,"8,607",293
"February 21, 2024",232,430,318,"8,573",231
"February 20, 2024",149,395,273,"8,488",192
"February 19, 2024",249,420,371,"8,380",218
"February 15, 2024",333,508,272,"8,463",312
"February 14, 2024",185,322,272,"8,512",230
"February 13, 2024",227,369,349,"8,348",160
"February 12, 2024",213,430,440,"8,092",234
"February 11, 2024",205,515,226,"8,056",347
"February 08, 2024",298,499,358,"8,188",315
"February 07, 2024",227,453,330,"8,160",222
"February 06, 2024",291,455,323,"8,045",178
"February 05, 2024",157,466,345,"7,872",242
"February 04, 2024",253,531,248,"7,813",342
"February 01, 2024",222,459,289,"8,026",337
"January 31, 2024",203,477,258,"8,153",266
"January 30, 2024",93,198,208,"8,182",187
"January 29, 2024",137,272,166,"8,057",195
"January 28, 2024",120,257,126,"8,004",307
"January 25, 2024",102,189,155,"8,180",293
"January 24, 2024",90,196,205,"8,212",243
"January 23, 2024",128,228,177,"8,133",202
"January 22, 2024",136,246,123,"8,008",222
"January 21, 2024",105,185,158,"7,974",371
"January 18, 2024",116,188,164,"8,042",359
"January 17, 2024",80,176,125,"8,072",353
"January 16, 2024",32,49,39,"8,133",188
"January 15, 2024",33,53,34,"8,033",234
"January 11, 2024",29,51,11,"8,829",476
"January 10, 2024",91,75,97,"9,144",349
"January 09, 2024",62,97,52,"9,311",286
"January 08, 2024",59,64,117,"9,670",367
"January 07, 2024",84,118,136,"9,670",367
"January 04, 2024",86,152,168,"10,558",362
"January 03, 2024",106,196,131,"10,718",263
"January 02, 2024",113,162,107,"10,770",316
"January 01, 2024",92,124,93,"
