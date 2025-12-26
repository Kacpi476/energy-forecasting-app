import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

from fetch_prices import fetch_prices
from fetch_pse import fetch_pse
from fetch_weather import fetch_weather

load_dotenv()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PRICES_FILE = DATA_DIR / "prices.parquet"
PSE_FILE = DATA_DIR / "pse.parquet"
WEATHER_FILE = DATA_DIR / "weather.parquet"


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
        start = pd.Timestamp("2024-07-01", tz="UTC")
        end = pd.Timestamp.now(tz="UTC").normalize()
        return start, end

    last = df.index.max()
    today = pd.Timestamp.now(tz="UTC").normalize()

    if last >= today:
        return None

    return last + pd.Timedelta(days=1), today


def update_file(path: Path, fetch_func):
    print(f"📂 Aktualizacja: {path}")

    if not path.exists():
        print("🆕 Plik nie istnieje – zapis pełny")
        df = fetch_func(
            pd.Timestamp("2024-07-01", tz="UTC"),
            pd.Timestamp.now(tz="UTC")
        )
        df.to_parquet(path)
        return

    df_old = pd.read_parquet(path)

    # 🔥 TUTAJ BYŁ CAŁY PROBLEM
    df_old = normalize_index(df_old)

    missing = get_missing_range(df_old)
    if not missing:
        print(f"✔ Plik {path.name} jest aktualny\n")
        return

    start, end = missing
    print(f"🔄 Pobieranie nowych danych od {start.date()} do {end.date()}")

    df_new = fetch_func(start, end)

    if df_new.empty:
        print("ℹ️ Brak nowych danych\n")
        return

    df = (
        pd.concat([df_old, df_new])
        .sort_index()
        .drop_duplicates()
    )

    df.to_parquet(path)
    print(f"✔ Zaktualizowano {path.name} (+{len(df_new)} rekordów)\n")


def main():
    update_file(PRICES_FILE, fetch_prices)
    update_file(PSE_FILE, fetch_pse)
    update_file(WEATHER_FILE, fetch_weather)
    print("✅ Wszystkie dane zaktualizowane!")


if __name__ == "__main__":
    main()
