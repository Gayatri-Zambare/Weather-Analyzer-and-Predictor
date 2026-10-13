
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Weather Analyzer and Predictor",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Weather Analyzer and Predictor")
st.write(
    "Analyze historical weather data and explore "
    "basic temperature predictions for Indian cities."
)

DATA_FILE = Path("data/weather_india.csv")

if not DATA_FILE.exists():
    st.error("Dataset not found. Please run download_data.py first.")
    st.stop()

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

cities = sorted(df["city"].dropna().unique())

st.sidebar.header("Dashboard Settings")
city = st.sidebar.selectbox("Select a city", cities)

city_df = df[df["city"] == city].copy()
city_df = city_df.sort_values("date")

st.header(f"Weather Overview: {city}")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Temperature",
    f"{city_df['mean_temp'].mean():.1f} °C"
)
col2.metric(
    "Average Daily Rainfall",
    f"{city_df['rainfall_mm'].mean():.1f} mm"
)
col3.metric(
    "Average Humidity",
    f"{city_df['humidity'].mean():.1f}%"
)
col4.metric(
    "Average Maximum Wind Speed",
    f"{city_df['wind_speed_max'].mean():.1f} km/h"
)

st.subheader("Historical Temperature Trend")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    city_df["date"],
    city_df["mean_temp"],
    linewidth=0.8
)
ax.set_xlabel("Date")
ax.set_ylabel("Mean Temperature (°C)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Monthly Weather Analysis")

city_df["year_month"] = city_df["date"].dt.to_period("M").astype(str)

monthly = city_df.groupby("year_month").agg(
    mean_temp=("mean_temp", "mean"),
    rainfall_mm=("rainfall_mm", "sum")
).reset_index()

weather_metric = st.selectbox(
    "Choose a measurement",
    ["mean_temp", "rainfall_mm"]
)

if weather_metric == "mean_temp":
    st.line_chart(monthly.set_index("year_month")["mean_temp"])
else:
    st.bar_chart(monthly.set_index("year_month")["rainfall_mm"])

st.subheader("Temperature Prediction")

st.caption(
    "This is a basic date-based machine learning demonstration, "
    "not a reliable operational weather forecast."
)

city_df["year"] = city_df["date"].dt.year
city_df["month"] = city_df["date"].dt.month
city_df["day_of_year"] = city_df["date"].dt.dayofyear

features = ["year", "month", "day_of_year"]
X = city_df[features]
y = city_df["mean_temp"]

if len(city_df) < 30:
    st.warning("Not enough records to train the demonstration model.")
else:
    model = LinearRegression()
    model.fit(X, y)

    min_date = city_df["date"].min().date()
    max_date = city_df["date"].max().date()

    prediction_date = st.date_input(
        "Choose a prediction date",
        value=pd.Timestamp("2026-01-15").date(),
        min_value=min_date,
        max_value=pd.Timestamp("2035-12-31").date()
    )

    date_value = pd.Timestamp(prediction_date)

    prediction_input = pd.DataFrame([{
        "year": date_value.year,
        "month": date_value.month,
        "day_of_year": date_value.dayofyear
    }])

    if prediction_date > max_date:
        st.info(
            "The selected date is after the historical dataset. "
            "This result is an extrapolation and may be inaccurate."
        )

    predicted_temp = model.predict(prediction_input)[0]

    st.metric(
        f"Estimated Mean Temperature for {city}",
        f"{predicted_temp:.1f} °C"
    )

st.subheader("Historical Weather Records")

st.write(f"Total records for {city}: {len(city_df):,}")

st.dataframe(
    city_df[
        [
            "date",
            "mean_temp",
            "max_temp",
            "min_temp",
            "rainfall_mm",
            "humidity",
            "wind_speed_max"
        ]
    ].sort_values("date", ascending=False),
    use_container_width=True
)

st.download_button(
    "Download Selected City's Data as CSV",
    data=city_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{city.lower()}_weather.csv",
    mime="text/csv"
)

st.caption("Scientific Computing with Python Mini Project")
