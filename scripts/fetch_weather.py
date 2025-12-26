# fetch_weather.py
import pandas as pd
from datetime import datetime, timedelta, timezone
import requests
from pathlib import Path

LAT = 52.2297
LON = 21.0122

def fetch_weather(start, end):
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={start_str}&end_date={end_str}"
        "&hourly=temperature_2m,wind_speed_10m,shortwave_radiation"
        "&timezone=UTC"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temperature_c": data["temperature_2m"],
        "wind_speed_ms": data["wind_speed_10m"],
        "solar_wm2": data["shortwave_radiation"]
    })
    df = df.set_index("date").sort_index()
    df.index = df.index.tz_localize("UTC")

    return df
