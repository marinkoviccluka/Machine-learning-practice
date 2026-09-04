import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os
import numpy as np

# Load data
fname = r"LV4\data_C02_emission.csv"
df = pd.read_csv(fname)

# Select only numerical columns without missing values
df_numeric = df.select_dtypes(['int64', 'float64'])

# X contains all numerical features except the target variable
X = df_numeric.drop(columns=["CO2 Emissions (g/km)"])
y = df["CO2 Emissions (g/km)"]

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Select engine size feature for inspection
feature = "Engine Size (L)" if "Engine Size (L)" in X.columns else X.columns[0]

os.makedirs("output", exist_ok=True)

plt.figure()
plt.scatter(X_train[feature], y_train, c="blue", label="train", alpha=0.5)
plt.scatter(X_test[feature], y_test, c="red", label="test", alpha=0.5)
plt.xlabel(feature)
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
plt.tight_layout()
plt.savefig("output/zad1_scatter_train_test.png", dpi=150)

# Feature scaling (normalization)
scaler = MinMaxScaler()
X_train_n = scaler.fit_transform(X_train)
X_test_n = scaler.transform(X_test)

# Histogram before and after scaling
orig = X_train[feature].values
index = list(X_train.columns).index(feature)
scaled = X_train_n[:, index]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Original
axes[0].hist(orig, bins=20, alpha=0.7, color="blue")
axes[0].set_title("Original")
axes[0].set_xlabel(feature)
axes[0].set_ylabel("count")

# Scaled
axes[1].hist(scaled, bins=20, alpha=0.7, color="green")
axes[1].set_title("Scaled")
axes[1].set_xlabel(feature)
axes[1].set_ylabel("count")

plt.tight_layout()
plt.savefig("output/zad1_hist_feature_scaling.png", dpi=150)

# Linear regression
lin = LinearRegression()
lin.fit(X_train_n, y_train)

print("Model parameters:")
print("theta (intercept):", lin.intercept_)
for name, coef in zip(X_train.columns, lin.coef_):
    print(f"theta for {name}: {coef}")

y_test_p = lin.predict(X_test_n)

plt.figure()

# True vs Predicted
plt.scatter(y_test, y_test_p, alpha=0.6, label="Predictions")

# Ideal model line (y = x)
min_val = min(y_test.min(), y_test_p.min())
max_val = max(y_test.max(), y_test_p.max())

plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal (y = x)")

plt.xlabel("Actual CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.legend()

plt.tight_layout()
plt.savefig("output/zad1_y_true_vs_pred.png", dpi=150)

# Evaluation metrics
mse = mean_squared_error(y_test, y_test_p)
mae = mean_absolute_error(y_test, y_test_p)
rmse = mse ** 0.5
mape = mean_absolute_percentage_error(y_test, y_test_p)
r2 = r2_score(y_test, y_test_p)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("MAE:", mae)
print("R2:", r2)