# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# 📥 Load CSV
df = pd.read_csv("NSE_BANK.csv")

# 🧹 Clean and prepare data
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# 🧭 Sidebar: Symbol selection
st.title("📈 ARIMA Forecasting for NSE Bank Stocks")
symbols = df['Symbol'].unique()
selected_symbol = st.sidebar.selectbox("Choose a stock symbol", symbols)

# 📊 Filter selected stock
st_data = df[df['Symbol'] == selected_symbol][['Close']].copy()
st_data['Return'] = st_data['Close'].pct_change()
st_data.dropna(inplace=True)

# 📉 ADF Test Function
def check_stationarity(series):
    result = adfuller(series.dropna())
    st.subheader("📊 ADF Test Results")
    st.write(f"**ADF Statistic:** {result[0]:.4f}")
    st.write(f"**p-value:** {result[1]:.4f}")
    if result[1] <= 0.05:
        st.success("✅ The series is stationary.")
    else:
        st.warning("⚠️ The series is not stationary.")

# 🔍 Run ADF test on original and differenced series
check_stationarity(st_data['Close'])
st_data['Close_Diff'] = st_data['Close'].diff()
check_stationarity(st_data['Close_Diff'])

# 🔮 Fit ARIMA model
model = ARIMA(st_data['Close'], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)

# 📅 Generate forecast dates
forecast_dates = pd.date_range(start=st_data.index[-1], periods=11, freq='B')[1:]

# 📈 Plot with Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=st_data.index,
    y=st_data['Close'],
    mode='lines',
    name='Actual Prices',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=forecast,
    mode='lines+markers',
    name='Forecasted Prices',
    line=dict(color='red', dash='dash'),
    marker=dict(size=6, symbol='circle')
))

fig.add_shape(
    type="line",
    x0=forecast_dates[0],
    y0=min(st_data['Close'].min(), forecast.min()),
    x1=forecast_dates[0],
    y1=max(st_data['Close'].max(), forecast.max()),
    line=dict(color="gray", width=1, dash="dot")
)

fig.update_traces(hovertemplate='Date: %{x}<br>Price: %{y:.2f} INR')
fig.update_layout(
    title=f"{selected_symbol} Stock Price Forecast",
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(x=0, y=1),
    yaxis=dict(range=[
        min(st_data['Close'].min(), forecast.min()) * 0.95,
        max(st_data['Close'].max(), forecast.max()) * 1.05
    ])
)

st.plotly_chart(fig, use_container_width=True)

# 📋 Forecast Table
forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Forecasted Price": forecast
})
st.subheader("📋 Forecasted Prices")
st.dataframe(forecast_df.style.format({"Forecasted Price": "{:.2f}"}))
