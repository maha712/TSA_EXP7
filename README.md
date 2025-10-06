# Ex.No: 07                                       AUTO REGRESSIVE MODEL

REG NO : 212222240057

NAME: MAHALAKSHMI K



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
# -------------------------------
# 1. Import necessary libraries
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# -------------------------------
# 2. Load dataset
# -------------------------------
df = pd.read_csv("/content/Netflix Dataset.csv")

# Convert Release_Date to datetime
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')

# Drop rows without valid dates
df = df.dropna(subset=['Release_Date'])

# Set index
df.set_index('Release_Date', inplace=True)

# Monthly count of releases
monthly_releases = df.resample("MS").size()
monthly_releases = monthly_releases.asfreq("MS")  # ensure continuous monthly index

# -------------------------------
# 3. Augmented Dickey-Fuller test
# -------------------------------
result = adfuller(monthly_releases.dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:", result[4])

if result[1] < 0.05:
    print("✅ Series is stationary")
else:
    print("⚠️ Series is NOT stationary")

# -------------------------------
# 4. Train-test split
# -------------------------------
train_size = int(len(monthly_releases) * 0.8)
train, test = monthly_releases.iloc[:train_size], monthly_releases.iloc[train_size:]

# -------------------------------
# 5. Fit AutoRegressive (AR) model
# -------------------------------
lag = 13
ar_model = AutoReg(train, lags=lag, old_names=False)
ar_fit = ar_model.fit()
print(ar_fit.summary())

# -------------------------------
# 6. Plot ACF and PACF
# -------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_acf(monthly_releases.dropna(), lags=40, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)")

plt.subplot(1, 2, 2)
plot_pacf(monthly_releases.dropna(), lags=40, ax=plt.gca())
plt.title("Partial Autocorrelation Function (PACF)")
plt.tight_layout()
plt.show()

# -------------------------------
# 7. Make predictions
# -------------------------------
start = len(train)
end = len(train) + len(test) - 1
predictions = ar_fit.predict(start=start, end=end, dynamic=False)

# -------------------------------
# 8. Evaluate with MSE
# -------------------------------
mse = mean_squared_error(test, predictions)
print("Mean Squared Error (MSE):", mse)

# -------------------------------
# 9. Plot Test vs Predictions
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual Test Data', color='blue')
plt.plot(test.index, predictions, label='AR Model Predictions', color='red', linestyle='--')
plt.title(f"AutoRegressive Model (lags={lag}) - Test vs Predictions")
plt.xlabel("Date")
plt.ylabel("Number of Netflix Releases")
plt.legend()
plt.grid()
plt.show()

```

### OUTPUT:

<img width="1416" height="692" alt="Screenshot (12)" src="https://github.com/user-attachments/assets/8d900cb3-d94b-4818-acb4-e369c05103ad" />


<img width="905" height="356" alt="Screenshot (13)" src="https://github.com/user-attachments/assets/e217afa4-121e-4637-9787-c4adff298528" />



<img width="1491" height="606" alt="Screenshot (14)" src="https://github.com/user-attachments/assets/94de53ce-ab91-405e-91cd-97d29371b925" />


<img width="1289" height="720" alt="Screenshot (15)" src="https://github.com/user-attachments/assets/179a6c5b-b727-46c7-a1ec-4b5f607907a9" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
