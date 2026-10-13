
import time
from pathlib import Path

import pandas as pd
import requests

# Dataset folder
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = DATA_DIR / "weather_india.csv"

# Five completely new Indian cities
CITIES = {
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Kolkata": (22.5726, 88.3639),
    "Lucknow": (26.8467, 80.9462),
    "Bhopal": (23.2599, 77.4126),
}

URL = "https://archive-api.open-meteo.com/v1/archive"

BASE_PARAMS = {
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "daily": (
        "temperature_2m_mean,"
        "temperature_2m_max,"
        "temperature_2m_min,"
        "precipitation_sum,"
        "relative_humidity_2m_mean,"
        "wind_speed_10m_max"
    ),
    "timezone": "Asia/Kolkata",
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
}

COLUMNS = [
    "date",
    "mean_temp",
    "max_temp",
    "min_temp",
    "rainfall_mm",
    "humidity",
    "wind_speed_max",
    "city",
]


def download_city(city, latitude, longitude):
    params = BASE_PARAMS.copy()
    params["latitude"] = latitude
    params["longitude"] = longitude

    for attempt in range(1, 4):
        try:
            print(f"\nDownloading {city} (attempt {attempt}/3)...")

            response = requests.get(
                URL,
                params=params,
                timeout=90
            )

            if response.status_code == 429:
                if attempt < 3:
                    wait_seconds = 60 * attempt
                    print(
                        f"Rate limit reached. "
                        f"Waiting {wait_seconds} seconds."
                    )
                    time.sleep(wait_seconds)
                    continue

                print(f"Failed to download {city}: rate limit.")
                return None

            response.raise_for_status()
            daily = response.json()["daily"]

            data = pd.DataFrame({
                "date": daily["time"],
                "mean_temp": daily["temperature_2m_mean"],
                "max_temp": daily["temperature_2m_max"],
                "min_temp": daily["temperature_2m_min"],
                "rainfall_mm": daily["precipitation_sum"],
                "humidity": daily["relative_humidity_2m_mean"],
                "wind_speed_max": daily["wind_speed_10m_max"],
                "city": city,
            })

            print(f"{city}: {len(data)} records downloaded.")
            return data

        except (requests.RequestException, KeyError, TypeError) as error:
            print(f"Error for {city}: {error}")
            if attempt < 3:
                time.sleep(10 * attempt)

    return None


def main():
    all_data = []

    for city, (latitude, longitude) in CITIES.items():
        data = download_city(city, latitude, longitude)

        if data is not None:
            all_data.append(data)

        # Pause between API requests
        time.sleep(5)

    if len(all_data) != len(CITIES):
        print(
            "\nSome cities failed to download. "
            "The dataset will contain only successful cities."
        )

    if not all_data:
        print("No data downloaded. Please try again later.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined[COLUMNS]
    combined["date"] = pd.to_datetime(combined["date"])

    # Replace the old dataset with the new five-city dataset
    combined.to_csv(CSV_FILE, index=False)

    print("\nDownload summary")
    print("------------------------------")
    print("Saved file:", CSV_FILE)
    print("Dataset shape:", combined.shape)

    print("\nRecords per city:")
    print(combined["city"].value_counts())

    print("\nMissing values:")
    print(combined.isnull().sum())

    print("\nFirst five rows:")
    print(combined.head())

    if set(combined["city"].unique()) == set(CITIES.keys()):
        print("\nSuccess! All five new cities are included.")
    else:
        missing = set(CITIES.keys()) - set(combined["city"].unique())
        print("\nStill missing:", ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
