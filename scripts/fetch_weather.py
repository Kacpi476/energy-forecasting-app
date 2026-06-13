import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

LAT = 52.2297
LON = 21.0122

def fetch_weather(start, end):
    now = datetime.now(timezone.utc)
    archive_cutoff = now - timedelta(days=5)  # bezpieczny bufor

    dfs = []

    # Część historyczna - przez /archive
    if start < archive_cutoff:
        archive_end = min(end, archive_cutoff)
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={LAT}&longitude={LON}"
            f"&start_date={start.strftime('%Y-%m-%d')}"
            f"&end_date={archive_end.strftime('%Y-%m-%d')}"
            "&hourly=temperature_2m,wind_speed_10m,shortwave_radiation"
            "&timezone=UTC"
        )
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()["hourly"]
        dfs.append(pd.DataFrame({
            "date": pd.to_datetime(data["time"]),
            "temperature_c": data["temperature_2m"],
            "wind_speed_ms": data["wind_speed_10m"],
            "solar_wm2": data["shortwave_radiation"]
        }))

    # Część przyszła - przez /forecast
    if end > archive_cutoff:
        forecast_start = max(start, archive_cutoff)
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={LAT}&longitude={LON}"
            f"&start_date={forecast_start.strftime('%Y-%m-%d')}"
            f"&end_date={end.strftime('%Y-%m-%d')}"
            "&hourly=temperature_2m,wind_speed_10m,shortwave_radiation"
            "&timezone=UTC"
        )
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()["hourly"]
        dfs.append(pd.DataFrame({
            "date": pd.to_datetime(data["time"]),
            "temperature_c": data["temperature_2m"],
            "wind_speed_ms": data["wind_speed_10m"],
            "solar_wm2": data["shortwave_radiation"]
        }))

    df = pd.concat(dfs).drop_duplicates(subset=["date"]).sort_values("date")
    df = df.set_index("date")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    return df