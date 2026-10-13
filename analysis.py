
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
df = pd.read_csv("data/weather_india.csv")
df["date"] = pd.to_datetime(df["date"])

# Create output folder for graphs
Path("outputs").mkdir(exist_ok=True)

print("WEATHER DATA ANALYSIS")
print("=" * 35)

# 1. Basic dataset information
print("\nDataset shape:", df.shape)
print("\nCities:", df["city"].unique())

# 2. Average temperature for each city
print("\nAverage temperature by city:")
print(df.groupby("city")["mean_temp"].mean().round(2))

# 3. Average rainfall for each city
print("\nAverage daily rainfall by city:")
print(df.groupby("city")["rainfall_mm"].mean().round(2))

# 4. Maximum temperature for each city
print("\nHighest recorded daily maximum temperature:")
print(df.groupby("city")["max_temp"].max())

# 5. Temperature comparison graph
df.groupby("city")["mean_temp"].mean().plot(
    kind="bar",
    title="Average Temperature by City",
    xlabel="City",
    ylabel="Temperature (°C)",
    rot=30
)
plt.tight_layout()
plt.savefig("outputs/average_temperature.png")
plt.show()

# 6. Daily temperature trend for one year
year_data = df[df["date"].dt.year == 2025]

for city in df["city"].unique():
    city_data = year_data[year_data["city"] == city]
    plt.plot(
        city_data["date"],
        city_data["mean_temp"],
        label=city
    )

plt.title("Daily Average Temperature Trends - 2025")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/temperature_trends_2025.png")
plt.show()

print("\nAnalysis completed successfully!")
print("Graphs saved in the outputs folder.")
