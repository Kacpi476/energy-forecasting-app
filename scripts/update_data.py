import pandas as pd
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

FINAL_DATA_PATH = DATA_DIR / "final_training_data.parquet"


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ujednolica indeks DataFrame do UTC — obsługuje indeks datowy i kolumnę 'date'."""
    if pd.api.types.is_datetime64_any_dtype(df.index):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")  # POPRAWKA: było dt.localize (błąd runtime)
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")
        return df

    return df


def update_file(path: Path, fetch_func, look_ahead_hours: int = 0):
    """
    Aktualizuje plik Parquet w trybie przyrostowym (incremental load).
    Pobiera tylko brakujący zakres czasowy od ostatniego rekordu do teraz + look_ahead_hours.
    """
    print(f"\n--- Sprawdzanie: {path.name} ---")

    now_utc = pd.Timestamp.now(tz="UTC")
    start_default = pd.Timestamp("2024-07-01", tz="UTC")
    end_target = now_utc + pd.Timedelta(hours=look_ahead_hours)

    if not path.exists():
        print(f"Tworzenie nowego pliku {path.name} od {start_default.date()}...")
        df = fetch_func(start_default, end_target)
        if df is not None and not df.empty:
            df = normalize_index(df)
            df.to_parquet(path)
            print(f"Zapisano {len(df)} rekordów.")
        return

    df_old = pd.read_parquet(path)
    df_old = normalize_index(df_old)
    last_available = df_old.index.max()

    if last_available < end_target - pd.Timedelta(hours=1):
        print(f"Pobieranie brakujących danych: {last_available} → {end_target}")
        df_new = fetch_func(last_available, end_target)

        if df_new is not None and not df_new.empty:
            df_new = normalize_index(df_new)
            df_combined = pd.concat([df_old, df_new])
            df_combined = (
                df_combined[~df_combined.index.duplicated(keep="last")]
                .sort_index()
            )
            df_combined.to_parquet(path)
            print(f"Zaktualizowano. Nowy zakres: {df_combined.index.min().date()} → {df_combined.index.max().date()}")
        else:
            print(f"Brak nowych danych dla {path.name} (API jeszcze nie opublikowało).")
    else:
        print(f"Aktualny. Ostatni rekord: {last_available}")


def run_full_pipeline():
    """Główny potok ETL: pobieranie → merge → prognoza."""

    print("=" * 50)
    print("ETAP 1: Pobieranie danych (ETL)")
    print("=" * 50)

    # Horyzonty look-ahead dobrane do dostępności poszczególnych API:
    # - pogoda: 72h (forecast API Open-Meteo)
    # - PSE: 48h (plan pracy na kolejne doby)
    # - ceny ENTSO-E: 36h (rynek dnia następnego D-1)
    # - CO2: 0h (dane historyczne, brak prognoz)
    update_file(WEATHER_FILE, fetch_weather, look_ahead_hours=72)
    update_file(PSE_FILE, fetch_pse, look_ahead_hours=48)
    update_file(PRICES_FILE, fetch_prices, look_ahead_hours=36)
    update_file(CO2_FILE, fetch_co2, look_ahead_hours=0)

    print("\n" + "=" * 50)
    print("ETAP 2: Integracja danych (Merge)")
    print("=" * 50)

    df_merged = merge_datasets()
    df_merged.to_parquet(FINAL_DATA_PATH, index=False)

    last_price = df_merged[df_merged["price_eur_mwh"].notna()]["date"].max()
    future_count = df_merged[df_merged["price_eur_mwh"].isna()].shape[0]
    print(f"Plik finalny gotowy. Ostatnia cena: {last_price} | Godzin do prognozy: {future_count}")

    print("\n" + "=" * 50)
    print("ETAP 3: Prognozowanie Day-Ahead")
    print("=" * 50)

    run_backtest()

    print("\nPotok zakończony pomyślnie.")


if __name__ == "__main__":
    run_full_pipeline()