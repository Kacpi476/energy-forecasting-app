import pandas as pd
import requests
from datetime import datetime, timezone

LAT = 52.2297
LON = 21.0122

def fetch_weather(start, end):
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    
    # Sprawdzamy, czy zakres dotyczy przyszłości
    now = datetime.now(timezone.utc)
    # Jeśli data końcowa jest późniejsza niż wczoraj, używamy forecast
    # (API archiwalne zazwyczaj kończy się 2 dni temu)
    is_future = end.date() >= now.date()

    if is_future:
        base_url = "https://api.open-meteo.com/v1/forecast"
        print("🌐 Używam endpointu FORECAST (dane bieżące/przyszłe)")
    else:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        print("🏛️ Używam endpointu ARCHIVE (dane historyczne)")

    url = (
        f"{base_url}?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={start_str}&end_date={end_str}"
        "&hourly=temperature_2m,wind_speed_10m,shortwave_radiation"
        "&timezone=UTC"
    )
    
    r = requests.get(url)
    
    if r.status_code != 200:
        print(f"❌ Błąd API ({r.status_code}): {r.text}")
        r.raise_for_status()
    
    data = r.json()["hourly"]
    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temperature_c": data["temperature_2m"],
        "wind_speed_ms": data["wind_speed_10m"],
        "solar_wm2": data["shortwave_radiation"]
    })
    
    df = df.set_index("date").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
        
    return df