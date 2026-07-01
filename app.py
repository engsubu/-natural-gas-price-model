import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Natural Gas Price Estimator", layout="wide")
st.title("JPMC Task 1: Natural Gas Price Analysis")

@st.cache_data
def load_and_train(path="data/gas_prices.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    df = df.set_index("Date").asfreq("M") # Ensure monthly frequency
    
    # Fit Holt-Winters with trend + additive seasonality = 12 months
    model = ExponentialSmoothing(
        df["Price"], trend="add", seasonal="add", seasonal_periods=12
    ).fit()
    return df, model

def estimate_price(target_date, model, df):
    last_date = df.index.max()
    # Number of months to forecast
    months_ahead = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
    if months_ahead <= 0:
        # If date is in historical range, just return actual/closest
        return float(model.fittedvalues.asof(target_date))
    forecast = model.forecast(steps=months_ahead)
    return float(forecast.iloc[-1])

# --- UI ---
try:
    df, model = load_and_train()
except FileNotFoundError:
    st.error("`data/gas_prices.csv` not found. Upload the file to a `data/` folder in your repo.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Historical + Forecast")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Price"], label="Historical")
    
    # Forecast next 12 months
    future_dates = pd.date_range(df.index.max() + relativedelta(months=1), periods=12, freq="M")
    forecast_vals = model.forecast(steps=12)
    ax.plot(future_dates, forecast_vals, label="Forecast 12M", linestyle="--")
    ax.legend()
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    st.pyplot(fig)

with col2:
    st.subheader("Estimate Price for Any Date")
    input_date = st.date_input(
        "Pick a date", 
        value=df.index.max() + relativedelta(months=6),
        min_value=df.index.min().date(),
        max_value=(df.index.max() + relativedelta(months=12)).date()
    )
    
    if st.button("Estimate"):
        est = estimate_price(pd.to_datetime(input_date), model, df)
        st.metric(label=f"Estimated Price on {input_date}", value=f"${est:.2f}")

st.subheader("Key Insight")
st.write("Model uses 12-month seasonality. Natural gas typically peaks in winter due to heating demand.")
