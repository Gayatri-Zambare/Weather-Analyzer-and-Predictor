
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the weather dataset
df = pd.read_csv("data/weather_india.csv")
df["date"] = pd.to_datetime(df["date"])

# Select the city for prediction
city_name = "Ahmedabad"

city_df = df[df["city"] == city_name].copy()
city_df = city_df.sort_values("date")

# Create features from the date
city_df["year"] = city_df["date"].dt.year
city_df["month"] = city_df["date"].dt.month
city_df["day_of_year"] = city_df["date"].dt.dayofyear

features = ["year", "month", "day_of_year"]

X = city_df[features]
y = city_df["mean_temp"]

# Split the data chronologically:
# First 80% for training, last 20% for testing
split_index = int(len(city_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train the machine learning model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict temperatures for the test data
predictions = model.predict(X_test)

# Display model results
print("=" * 45)
print("WEATHER TEMPERATURE PREDICTION")
print("=" * 45)

print("Selected city:", city_name)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Mean Absolute Error:",
    round(mean_absolute_error(y_test, predictions), 2),
    "°C"
)

print("R-squared score:", round(r2_score(y_test, predictions), 3))

# Create the output folder
Path("outputs").mkdir(exist_ok=True)

# Plot actual and predicted temperatures
plt.figure(figsize=(12, 5))

plt.plot(
    city_df["date"].iloc[split_index:],
    y_test,
    label="Actual Temperature"
)

plt.plot(
    city_df["date"].iloc[split_index:],
    predictions,
    label="Predicted Temperature"
)

plt.title(f"Actual vs Predicted Temperature - {city_name}")
plt.xlabel("Date")
plt.ylabel("Mean Temperature (°C)")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/prediction_comparison.png")
plt.show()

print("\nGraph saved to outputs/prediction_comparison.png")

# Demonstration prediction for a future date
future_date = pd.Timestamp("2026-01-15")

future_features = pd.DataFrame([{
    "year": future_date.year,
    "month": future_date.month,
    "day_of_year": future_date.dayofyear
}])

future_prediction = model.predict(future_features)[0]

print(
    f"\nDemonstration prediction for {city_name} "
    f"on {future_date.date()}: {future_prediction:.2f} °C"
)

print(
    "Note: This is a basic date-based model, "
    "not a real weather forecast."
)
