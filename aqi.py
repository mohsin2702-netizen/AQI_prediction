import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("aqi_data.csv")
print(df.head())
print(df.shape)

# Drop rows where AQI is missing
df = df.dropna(subset=["AQI"])

# Fill missing values in feature columns
df["PM2.5"] = df["PM2.5"].fillna(df["PM2.5"].median())
df["PM10"] = df["PM10"].fillna(df["PM10"].median())
df["NO"] = df["NO"].fillna(df["NO"].median())
df["NO2"] = df["NO2"].fillna(df["NO2"].median())
df["CO"] = df["CO"].fillna(df["CO"].median())
df["SO2"] = df["SO2"].fillna(df["SO2"].median())
df["O3"] = df["O3"].fillna(df["O3"].median())

# Plot AQI distribution
plt.figure(figsize=(8,5))
sns.histplot(df["AQI"], bins=50, kde=True, color="steelblue")
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.savefig("aqi_plot.png")
plt.close()
print("Plot saved!")

# Features and target
features = ["PM2.5", "PM10", "NO", "NO2", "CO", "SO2", "O3"]
X = df[features]
y = df["AQI"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))
print("Linear Regression R2:", r2_score(y_test, lr_pred))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))

# Save model
joblib.dump(rf, "aqi_model.pkl")
print("Model saved!")