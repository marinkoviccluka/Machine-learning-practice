import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error, mean_absolute_percentage_error
import numpy as np

# Load data
fname = r"LV4\data_C02_emission.csv"
df = pd.read_csv(fname)

# Numerical columns excluding CO2
num_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
if "CO2 Emissions (g/km)" in num_cols:
    num_cols.remove("CO2 Emissions (g/km)")

# Categorical column: include Fuel Type if present, otherwise empty list
cat_cols = ["Fuel Type"] if "Fuel Type" in df.columns else []

# Feature matrix (X) and target vector (y)
X = df[num_cols + cat_cols]
y = df["CO2 Emissions (g/km)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# One-hot encoding for Fuel Type; remaining columns pass through without transformation
preprocess = ColumnTransformer(
    [("cat", OneHotEncoder(drop=None, sparse_output=False), cat_cols)],
    remainder="passthrough"
)

# Preprocessing and model pipeline
model = Pipeline([
    ("pre", preprocess),
    ("lin", LinearRegression())
])

# Training
model.fit(X_train, y_train)

# Prediction
y_test_p = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_test_p)
mae = mean_absolute_error(y_test, y_test_p)
rmse = mse ** 0.5
mape = mean_absolute_percentage_error(y_test, y_test_p)
r2 = r2_score(y_test, y_test_p)
mx = max_error(y_test, y_test_p)

print("MODEL METRICS")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("MAE:", mae)
print("R2:", r2)
print("Max error:", mx)
print("\nWORST ERROR ANALYSIS")

# Sample index with the largest absolute error
idx = np.argmax(abs(y_test - y_test_p))

# In pandas, iloc returns a row by its positional index in the DataFrame
true_val = y_test.iloc[idx]
pred_val = y_test_p[idx]
row_X = X_test.iloc[idx]

print(f"Actual CO2 emission: {true_val} g/km")
print(f"Predicted CO2 emission: {pred_val} g/km")
print(f"Model error for this vehicle: {abs(pred_val - true_val)} g/km")
print("\nInput features of the vehicle with the largest error:")
print(row_X)