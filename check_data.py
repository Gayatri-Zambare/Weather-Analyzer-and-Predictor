
import pandas as pd

df = pd.read_csv("data/weather_india.csv")

print("Dataset shape:", df.shape)
print("\nCities:")
print(df["city"].unique())

print("\nRecords per city:")
print(df["city"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
