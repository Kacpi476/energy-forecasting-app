"""
add_real_prices.py
------------------
Pobiera rzeczywiste ceny z ENTSO-E i wpisuje je do forecast_history.parquet
obok predicted_price. Logika pobierania identyczna jak w fetch_prices.py
i update_data.py.

Użycie:
    python scripts/add_real_prices.py
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
import traceback
import os

load_dotenv()
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

DATA_DIR      = Path("data")
FORECAST_PATH = DATA_DIR / "forecast_history.parquet"


def fetch_real_prices(start, end):
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

    start = pd.Timestamp(start).tz_convert("Europe/Brussels")
    end   = pd.Timestamp(end).tz_convert("Europe/Brussels")

    if start >= end:
        return pd.DataFrame(columns=["price_eur_mwh"])

    prices = client.query_day_ahead_prices(country_code="PL", start=start, end=end)

    df = prices.to_frame(name="price_eur_mwh")
    df.index = df.index.tz_convert("UTC")

    if "price" in df.columns:
        df = df.drop(columns=["price"], errors="ignore")

    df["date"] = df.index
    df = df.set_index("date").sort_index()

    return df


def add_real_prices():
    if not FORECAST_PATH.exists():
        print("Brak pliku forecast_history.parquet. Uruchom najpierw update_data.py.")
        return

    df = pd.read_parquet(FORECAST_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Godziny z prognozą ale bez ceny realnej
    missing = df[df["price_eur_mwh"].isna() & df["predicted_price"].notna()]

    if missing.empty:
        print("Brak godzin do uzupełnienia — wszystkie prognozy mają już ceny realne.")
        return

    start = missing["date"].min()

    # End: koniec następnego dnia od teraz (tak jak update_data z look_ahead_hours=36)
    # ENTSO-E publikuje ceny na cały dzień D+1 naraz — zaokrąglamy do końca dnia
    now_utc = pd.Timestamp.now(tz="UTC")
    end = (now_utc + pd.Timedelta(hours=36)).floor("D") + pd.Timedelta(hours=23)

    print(f"Godzin do uzupełnienia : {len(missing)}")
    print(f"Zakres pobierania      : {start} → {end}")

    try:
        real = fetch_real_prices(start, end)
    except Exception:
        print("Błąd pobierania cen z ENTSO-E:")
        traceback.print_exc()
        return

    if real is None or real.empty:
        print("ENTSO-E nie zwróciło danych — ceny mogą jeszcze nie być opublikowane.")
        return

    print(f"Pobrano {len(real)} rekordów: {real.index.min()} → {real.index.max()}")

    real = real.reset_index()
    real["date"] = pd.to_datetime(real["date"], utc=True)
    real = real.rename(columns={"price_eur_mwh": "price_real"})

    df = df.merge(real[["date", "price_real"]], on="date", how="left")
    mask = df["price_eur_mwh"].isna() & df["price_real"].notna()
    df.loc[mask, "price_eur_mwh"] = df.loc[mask, "price_real"]
    df.drop(columns=["price_real"], inplace=True)

    df.to_parquet(FORECAST_PATH, index=False)
    print(f"Uzupełniono {mask.sum()} godzin cenami realnymi.")

    # Podsumowanie błędu dla uzupełnionych godzin
    compare = df[
        df["price_eur_mwh"].notna() &
        df["predicted_price"].notna() &
        (df["date"] >= start)
    ].copy()

    if compare.empty:
        print("Brak wspólnych rekordów do porównania.")
        return

    compare["blad"] = (compare["price_eur_mwh"] - compare["predicted_price"]).abs()
    mae  = compare["blad"].mean()
    mape = (compare["blad"] / compare["price_eur_mwh"].replace(0, pd.NA)).mean() * 100

    print(f"\nWyniki weryfikacji prognozy ({len(compare)}h):")
    print(f"  MAE  : {mae:.2f} EUR/MWh")
    print(f"  MAPE : {mape:.1f}%")
    print(f"\n{'Data (UTC)':<32} {'Realna':>10} {'Prognoza':>10} {'Błąd':>10}")
    print("-" * 68)
    for _, row in compare.iterrows():
        print(f"{str(row['date']):<32} {row['price_eur_mwh']:>10.2f} "
              f"{row['predicted_price']:>10.2f} {row['blad']:>10.2f}")


if __name__ == "__main__":
    add_real_prices()