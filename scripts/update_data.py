import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

from fetch_prices import fetch_prices
from fetch_pse import fetch_pse
from fetch_weather import fetch_weather
from fetch_co2 import fetch_co2

from merge_data import merge_datasets
from forecast_engine import generate_forecasts

load_dotenv()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PRICES_FILE = DATA_DIR / "prices.parquet"
PSE_FILE = DATA_DIR / "pse.parquet"
WEATHER_FILE = DATA_DIR / "weather.parquet"
CO2_FILE = DATA_DIR / "co2.parquet"


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_datetime64_any_dtype(df.index):
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")
        return df

    if "dtime_utc" in df.columns:
        df["dtime_utc"] = pd.to_datetime(df["dtime_utc"], utc=True)
        df = df.set_index("dtime_utc")
        df.index.name = "date"
        return df

    raise ValueError("Brak kolumny datetime (date / dtime_utc)")


def get_missing_range(df: pd.DataFrame, look_ahead_hours=0):
    """
    Zwraca zakres od ostatniego rekordu do 'teraz' + look_ahead_hours.
    """
    start_date = pd.Timestamp("2024-07-01", tz="UTC")
    
    end_date = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=look_ahead_hours)

    if df.empty:
        return start_date, end_date

    last = df.index.max()
    if last.tz is None:
        last = last.tz_localize('UTC')
        
    if (end_date - last) > pd.Timedelta(hours=1):
        return last, end_date
    
    return None

def update_file(path: Path, fetch_func, look_ahead_hours=0):
    print(f"\n--- Sprawdzanie: {path.name} ---")
    
    start_default = pd.Timestamp("2024-07-01", tz="UTC")
    end_default = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=look_ahead_hours)

    if not path.exists():
        print(f"Tworzenie nowego pliku {path.name}...")
        df = fetch_func(start_default, end_default)
        if df is not None and not df.empty:
            df.to_parquet(path)
        return

    df_old = pd.read_parquet(path)
    df_old = normalize_index(df_old)

    missing = get_missing_range(df_old, look_ahead_hours=look_ahead_hours)
    
    if not missing:
        print(f"{path.name} jest aktualny (Ostatni rekord: {df_old.index.max()})")
        return

    start, end = missing
    print(f"Pobieranie brakujących danych: od {start} do {end}")

    df_new = fetch_func(start, end)

    if df_new is None or df_new.empty:
        print(f"Brak nowych danych dla {path.name} (możliwy brak publikacji na ten zakres)")
        return

    df_combined = pd.concat([df_old, df_new])
    if df_combined.index.tz is None:
        df_combined.index = df_combined.index.tz_localize('UTC')
    
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()

    df_combined.to_parquet(path)
    print(f"Zaktualizowano! Nowy zakres do: {df_combined.index.max()}")


def run_full_pipeline():
    update_file(PRICES_FILE, fetch_prices, look_ahead_hours=24)
    
    update_file(PSE_FILE, fetch_pse, look_ahead_hours=36)
    
    update_file(WEATHER_FILE, fetch_weather, look_ahead_hours=48)
    
    update_file(CO2_FILE, fetch_co2, look_ahead_hours=0)
    
    print("\n--- Wszystkie dane surowe zaktualizowane! ---")
    print("Rozpoczynam integrację danych (merging)...")
    
    df = merge_datasets()
    
    FINAL_DATA_PATH = DATA_DIR / "final_training_data.parquet"
    df.to_parquet(FINAL_DATA_PATH, index=False)
    
    print(f"Plik finalny zapisany: {FINAL_DATA_PATH}")
    print(f"Zakres danych wejściowych: {df['date'].min()} do {df['date'].max()}")

    print("\n--- Uruchamiam silnik prognozowania ---")
    forecasts = generate_forecasts()
    
    if forecasts is not None:
        print(f"✨ System gotowy. Najdalsza prognoza: {forecasts['date'].max()}")

if __name__ == "__main__":
    run_full_pipeline()