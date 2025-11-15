import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
df= pd.read_csv('NSE_BANK.csv')
df.columns = df.columns.get_level_values(0)
df.columns=df.reset_index(0)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
print(df.Symbol.unique())
a= input('Enter the Symbol:')
st= df[df.Symbol==a]
st=st[['Close']]
st['Return']= st['Close'].pct_change()
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings(action="ignore")
def check(t):
    result= adfuller(t.dropna())
    
    #These lines use f-strings (formatted string literals) 
    #to print specific values from the result object returned by the adfuller() function.
    
    print(f"AD Fuller: {result[0]}")
    print(f"P-Value: {result[1]}")
    if result[1] <= 0.05:
        print("The series is Stationary")
    else:
        print("The series is not Stationary")
check(st['Close'])
st['Close_Diff']= st['Close'].diff().dropna() # Nan value exists on checking st
check(st['Close_Diff'])
model= ARIMA(st['Close'],order=(5,1,0))
model_fit = model.fit()
forecast= model_fit.forecast(steps=10)
# Ensure forecast_dates is a proper datetime index
dates = pd.date_range(start=st.index[-1], periods=11, freq='B')[1:]
#dates = pd.to_datetime(dates)  # <-- This ensures it's datetime format

plt.figure(figsize=(10,5))
plt.plot(st['Close'], label="Actual Prices")
plt.plot(dates, forecast, label="Predicted Prices", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
