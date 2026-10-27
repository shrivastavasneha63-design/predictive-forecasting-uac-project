Predictive Forecasting of Care Load & Placement Demand

HHS Unaccompanied Children (UAC) Program

📌 Project Overview

This project develops a predictive forecasting framework to estimate future care load and placement demand within the HHS Unaccompanied Children (UAC) Program.

The project uses historical program data to identify trends in:

- Children in HHS Care
- Children discharged from HHS Care
- Care-load patterns
- Discharge and placement demand

Time-series forecasting techniques are used to generate short-term forecasts that can support capacity planning, resource allocation, staffing decisions, and operational preparedness.

🎯 Objectives

1. Analyze historical UAC care and discharge trends.
2. Prepare and clean the dataset for time-series analysis.
3. Develop short-term forecasting models.
4. Compare forecasting model performance.
5. Evaluate models using MAE, RMSE, and MAPE.
6. Generate 30-day forecasts for care load and discharge demand.
7. Present the results through an interactive Streamlit dashboard.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Time-Series Forecasting

📊 Forecasting Models

The project compares multiple forecasting approaches:

- Naive Forecast
- 7-Day Moving Average
- Exponential Smoothing

Model performance is evaluated using:

MAE — Mean Absolute Error
RMSE — Root Mean Squared Error
MAPE — Mean Absolute Percentage Error

📈 Dashboard

The Streamlit application provides:

- Historical care-load visualization
- Historical discharge-demand visualization
- Forecasting model comparison
- 7–30 day forecasting
- Forecast tables
- Approximate uncertainty bands
- Operational demand insights

📁 Project Structure

predictive-forecasting-uac-project/
│
├── app.py
├── HHS_Unaccompanied_Alien_Children_Program_cleaned.csv
├── requirements.txt
├── README.md
├── forecast_30_day_care.csv
├── forecast_30_day_discharge.csv
└── model_comparison.csv

▶️ How to Run

Install the required Python packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

📌 Expected Outcome

The project provides a data-driven approach for anticipating changes in UAC care load and discharge demand. The resulting forecasts can help demonstrate how predictive analytics may support proactive operational and placement-capacity planning.

---

Project: Predictive Forecasting of Care Load & Placement Demand
Program: HHS Unaccompanied Children (UAC) Program
