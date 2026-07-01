import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Natural Gas Price Model", layout="wide")
st.title("📈 JPMorgan Task 1: Natural Gas Price Analysis")
st.write("Holt-Winters model with 12-month seasonality to forecast future prices.")

@st.cache_data
def load_and_train():
    """Load CSV, clean it, and fit Holt-Winters model"""
    path = "data/gas_prices.csv"
    
    # 1. Load with YOUR column names: Dates, Prices
    df = pd.read_csv(path)
    df = df.rename(columns={"Dates": "Date", "Prices": "Price"})
    
    # 2. Clean data: handle 10/31/20 format + 1.01E+01 scientific notation
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna().sort_values("Date")
    
    # 3. Set monthly frequency. Pandas 2.0+ uses 'ME' not 'M'
    df = df.set_index("Date").asfreq("ME") 
    df["Price"] = df["Price"].interpolate()
    
    # 4. Fit Holt-Winters: additive trend + additive seasonality, period=12
    model = ExponentialSmoothing(
        df["Price"], 
        trend="add", 
        seasonal="add", 
        seasonal_periods=12
    ).fit(optimized=True)
    
    return df, model

try:
    df, model = load_and_train()
except FileNotFoundError:
    st.error("`data/gas_prices.csv` not found. Make sure the file is in a `data/` folder on GitHub.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar inputs
st.sidebar.header("Forecast Settings")
months_ahead = st.sidebar.slider("Months to Forecast", 1, 24, 12)

# Forecast
forecast = model.forecast(months_ahead)
last_date = df.index[-1]
# Pandas 2.0+ uses 'ME' not 'M'
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=months_ahead, freq="ME") 
forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast.values})

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["Price"], label="Historical Price", linewidth=2)
ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", linestyle="--", linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Price $/MMBtu")
ax.set_title("Natural Gas Price: Historical + Forecast")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# Table + Metrics
st.subheader(f"Next {months_ahead} Month Forecast")
forecast_df["Date"] = forecast_df["Date"].dt.strftime("%b %Y")
forecast_df["Forecast"] = forecast_df["Forecast"].round(2)
st.dataframe(forecast_df, use_container_width=True, hide_index=True)

col1, col2, col3 = st.columns(3)
col1.metric("Last Actual Price", f"${df['Price'].iloc[-1]:.2f}")
col2.metric(f"{months_ahead}M Forecast", f"${forecast.iloc[-1]:.2f}")
col3.metric("Avg Forecast", f"${forecast.mean():.2f}")

st.caption("Model: Holt-Winters Exponential Smoothing with 12-month seasonality")
