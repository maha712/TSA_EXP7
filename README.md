# Ex.No: 07                                       AUTO REGRESSIVE MODEL

REG NO : 212222240057

NAME: MAHALAKSHMI K

### Date: 

### AIM:

To Implementat an Auto Regressive Model using Python

### ALGORITHM:

1. Import necessary libraries

2. Read the CSV file into a DataFrame

3. Perform Augmented Dickey-Fuller test

4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags

5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)

6. Make predictions using the AR model.Compare the predictions with the test data

7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.

### PROGRAM
```
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Sample data creation for demonstration
data = {
    'Date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'Cases': [10, 15, 20, 30, 50, 70, 100, 150, 200, 250] * 10
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Fit the Auto-Regressive model
ar_model = AutoReg(df['Cases'], lags=1)
ar_fit = ar_model.fit()
df['AR_Predictions'] = ar_fit.predict(start=len(df), end=len(df)+29, dynamic=False)

# Fit the Exponential Smoothing model
es_model = ExponentialSmoothing(df['Cases'], trend='add', seasonal='add', seasonal_periods=7)
es_fit = es_model.fit()
df['ES_Predictions'] = es_fit.fittedvalues

# Forecasting
ar_forecast = ar_fit.predict(start=len(df), end=len(df)+29, dynamic=False)
es_forecast = es_fit.forecast(steps=30)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(df['Cases'], label='Actual Cases', color='blue')
plt.plot(df['AR_Predictions'], label='AR Model Predictions', color='orange')
plt.plot(df['ES_Predictions'], label='Exponential Smoothing Fit', color='green')
plt.plot(pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30), ar_forecast, label='AR Forecast', color='red', linestyle='--')
plt.plot(pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30), es_forecast, label='ES Forecast', color='purple', linestyle='--')
plt.title('COVID-19 Cases Forecasting with AR Model and Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.grid()
plt.show()
```

### OUTPUT:

![376948964-1b25c294-a409-4ced-909d-fabb6352f96c](https://github.com/user-attachments/assets/15aa55db-6e67-4c50-8947-adae4434fcaa)


![376949072-469de4fa-15a9-4d62-b663-2c7afa827e56](https://github.com/user-attachments/assets/e95a7a49-eb6a-4ad9-bac7-9d6db37cda52)

![376949139-caaec05a-7259-4eb0-bd23-123e5849bd04](https://github.com/user-attachments/assets/f99185fe-d620-4e1b-a5d6-a61bf7fee13c)

![376949401-2ad6fa22-7ff0-45b0-a0e4-5b7dbb8ac8b8](https://github.com/user-attachments/assets/631377cb-ca9d-4c2b-8f78-db1e3a5c8408)

![376949466-144c35de-abf8-4dd8-9af2-ad73e88c21db](https://github.com/user-attachments/assets/69f77208-cd11-4194-93e5-11991ac82e6e)

![376949640-d5adcb02-2712-4025-8312-1ff8b389b34e](https://github.com/user-attachments/assets/81a3ef1b-704c-4dfb-87c9-1226b4599153)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
