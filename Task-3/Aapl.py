import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12,6)

# Load Apple stock data
df = pd.read_csv(r"B:\internship offer letter\Task-3\AAPL.csv", parse_dates=["Date"], index_col="Date")

print("Data Shape:", df.shape)
print(df.head())

# We'll use 'Close' price for time series
ts = df["Close"]
ts.plot(title="Apple Stock Closing Price")
plt.show()

# Decompose (multiplicative model for stock patterns)
decomposition = seasonal_decompose(ts, model="multiplicative", period=30)  # 30 ~ monthly cycle
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

df["MA_20"] = ts.rolling(window=20).mean()
df["MA_50"] = ts.rolling(window=50).mean()

plt.plot(df["Close"], label="Close")
plt.plot(df["MA_20"], label="20-Day MA", linestyle="--")
plt.plot(df["MA_50"], label="50-Day MA", color="red")
plt.legend()
plt.title("Apple Stock Moving Averages")
plt.show()

# Use last 60 days as test
train = ts.iloc[:-60]
test = ts.iloc[-60:]
print("Train size:", train.shape, "| Test size:", test.shape)

# Fit ARIMA model
model = ARIMA(train, order=(5,1,2))  # Example order, can be tuned with AIC/BIC
result = model.fit()

# Forecast next 60 days
forecast = result.forecast(steps=60)

# Evaluate RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"RMSE: {rmse:.2f}")

# Plot forecast vs actual
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual", color="blue")
plt.plot(test.index, forecast, label="Forecast", color="red", linestyle="--")
plt.fill_between(test.index, forecast*0.9, forecast*1.1, alpha=0.2)
plt.title(f"ARIMA Forecast (RMSE={rmse:.2f})")
plt.legend()
plt.show()
