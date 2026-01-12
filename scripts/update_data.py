#update_data.py

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
    """
    ⚠️ KLUCZOWA FUNKCJA
    Jeśli indeks nie jest datetime:
    - próbuje użyć kolumny 'date'
    - albo 'dtime_utc'
    """
    if pd.api.types.is_datetime64_any_dtype(df.index):
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

    raise ValueError("❌ Brak kolumny datetime (date / dtime_utc)")


def get_missing_range(df: pd.DataFrame):
    if df.empty:
        return pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp.now(tz="UTC")

    last = df.index.max()
    # Upewnienie się, że mamy strefę czasową
    if last.tz is None:
        last = last.tz_localize('UTC')
        
    now = pd.Timestamp.now(tz="UTC")

    # Jeśli różnica między 'teraz' a ostatnim rekordem jest większa niż 1 godzina
    if (now - last) > pd.Timedelta(hours=1):
        # Pobieraj od ostatniego punktu do teraz
        return last, now
    
    return None

def update_file(path: Path, fetch_func):
    print(f"📂 Sprawdzanie: {path.name}")
    
    if not path.exists():
        df = fetch_func(pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp.now(tz="UTC"))
        if df is not None and not df.empty:
            df.to_parquet(path)
        return

    df_old = pd.read_parquet(path)
    df_old = normalize_index(df_old)

    missing = get_missing_range(df_old)
    print(f"DEBUG: Zakres brakujący dla {path.name}: {missing}")
    if not missing:
        print(f"✔ {path.name} jest aktualny (Ostatni rekord: {df_old.index.max()})")
        return

    start, end = missing
    # Dodajemy mały margines, żeby nie dublować, ale też nic nie pominąć
    print(f"🔄 Wymuszam pobieranie dla {path.name}: od {start} do {end}")

    df_new = fetch_func(start, end)

    if df_new is None or df_new.empty:
        print(f"⚠️ Serwer nie zwrócił nowych danych dla {path.name} (możliwy brak publikacji)")
        return

    # Łączenie i usuwanie duplikatów po czasie
    df_combined = pd.concat([df_old, df_new])
    # Kluczowe: sortujemy i bierzemy ostatnie wystąpienie danej godziny
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()

    df_combined.to_parquet(path)
    print(f"✨ Zaktualizowano! Nowy koniec: {df_combined.index.max()}")


def run_full_pipeline():
    # 1. Pobieranie nowych danych do plików cząstkowych
    update_file(PRICES_FILE, fetch_prices)
    update_file(PSE_FILE, fetch_pse)
    update_file(WEATHER_FILE, fetch_weather)
    update_file(CO2_FILE, fetch_co2)
    
    print("\nWszystkie dane surowe zaktualizowane!")
    print("Rozpoczynam mergowanie danych...")
    
    # 2. Wywołanie merge_datasets
    # Funkcja merge_datasets (którą masz w osobnym pliku) zwraca gotowy DataFrame
    df = merge_datasets()
    
    # 3. ZAPIS do pliku finalnego (kluczowy krok dla aplikacji i modelu)
    # Zapisujemy bez indeksu, bo Twoja funkcja merge_datasets robi reset_index()
    FINAL_DATA_PATH = DATA_DIR / "final_training_data.parquet"
    df.to_parquet(FINAL_DATA_PATH, index=False)
    
    print(f"Plik zmergowany zapisany pomyślnie: {FINAL_DATA_PATH}")
    print(f"Zakres danych: {df['date'].min()} do {df['date'].max()}")
    print(f"Gotowy do prognozowania!")

    print("Uruchamiam system predykcji...")
    forecasts = generate_forecasts()
    
    # Opcjonalnie: zapisz ostatnią prognozę do JSON dla strony www
    if forecasts is not None:
        latest = forecasts.tail(24) # Prognoza na najbliższą dobę
        latest.to_json("data/latest_forecast.json", date_format='iso')
        print("🚀 System gotowy. Nowa prognoza zapisana!")

if __name__ == "__main__":
    run_full_pipeline()
