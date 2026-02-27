import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

from fetch_prices import fetch_prices
from fetch_pse import fetch_pse
from fetch_weather import fetch_weather
from fetch_co2 import fetch_co2

from merge_data import merge_datasets
from backtest_engine import run_backtest 

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
    return df

def update_file(path: Path, fetch_func, look_ahead_hours=0):
    print(f"\n--- Sprawdzanie: {path.name} ---")
    
    now_utc = pd.Timestamp.now(tz="UTC")
    start_default = pd.Timestamp("2024-07-01", tz="UTC")

    end_target = now_utc + pd.Timedelta(hours=look_ahead_hours)

    if not path.exists():
        print(f"Tworzenie nowego pliku {path.name}...")
        df = fetch_func(start_default, end_target)
        if df is not None and not df.empty:
            df.to_parquet(path)
        return

    df_old = pd.read_parquet(path)
    df_old = normalize_index(df_old)
    last_available = df_old.index.max()

    if last_available < end_target - pd.Timedelta(hours=1):
        print(f"Pobieranie brakujących danych: od {last_available} do {end_target}")
        df_new = fetch_func(last_available, end_target)
        
        if df_new is not None and not df_new.empty:
            df_new = normalize_index(df_new)
            df_combined = pd.concat([df_old, df_new])
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()
            df_combined.to_parquet(path)
            print(f"Zaktualizowano! Nowy zakres do: {df_combined.index.max()}")
        else:
            print(f"Brak nowych danych dla {path.name} (Giełda/API jeszcze nie opublikowało).")
    else:
        print(f"{path.name} jest aktualny (Ostatni rekord: {last_available})")

def run_full_pipeline():
    #POBIERANIE DANYCH
    update_file(PRICES_FILE, fetch_prices, look_ahead_hours=36)
    update_file(PSE_FILE, fetch_pse, look_ahead_hours=48)
    update_file(WEATHER_FILE, fetch_weather, look_ahead_hours=72)
    update_file(CO2_FILE, fetch_co2, look_ahead_hours=0)
    
    print("\n---(Merging) ---")
    df_merged = merge_datasets()
    
    FINAL_DATA_PATH = DATA_DIR / "final_training_data.parquet"
    df_merged.to_parquet(FINAL_DATA_PATH, index=False)
    
    print(f"Plik finalny gotowy. Max data w danych: {df_merged['date'].max()}")

    print("\n--- Silnik Prognozowania (Day-Ahead 24h) ---")
    run_backtest()

if __name__ == "__main__":
    run_full_pipeline()