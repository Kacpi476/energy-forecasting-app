"""
fill_missing_prices.py
----------------------
Sprawdza plik data/prices.parquet i pobiera z ENTSO-E wszystkie
brakujące godziny – zarówno luki w środku historii, jak i brak danych
na końcu (aż do ostatnio opublikowanych cen Day-Ahead).

Użycie:
    python fill_missing_prices.py

Logika:
  1. Wczytuje prices.parquet (jeśli nie istnieje – tworzy od START_DATE).
  2. Buduje pełny godzinowy indeks UTC od START_DATE do teraz+36h.
  3. Wykrywa wszystkie luki (NaN lub brakujące wiersze).
  4. Grupuje luki w ciągłe przedziały i odpytuje ENTSO-E osobno dla każdego.
  5. Scala wyniki z istniejącym plikiem i zapisuje.
"""

import os
import traceback

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from pathlib import Path

load_dotenv()

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
DATA_DIR   = Path("data")
PRICES_FILE = DATA_DIR / "prices.parquet"
START_DATE  = pd.Timestamp("2024-07-01", tz="UTC")

# ENTSO-E publikuje ceny D+1 — nie odpytujemy dalej niż teraz+36h
def get_end_target():
    return pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=36)


def fetch_prices_range(client: EntsoePandasClient, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Pobiera ceny Day-Ahead PL dla podanego zakresu. Zwraca DataFrame z indeksem UTC."""
    try:
        s = start.tz_convert("Europe/Brussels")
        e = end.tz_convert("Europe/Brussels")
        prices = client.query_day_ahead_prices(country_code="PL", start=s, end=e)
        df = prices.to_frame(name="price_eur_mwh")
        df.index = df.index.tz_convert("UTC")
        df.index.name = "date"
        return df.sort_index()
    except Exception as exc:
        print(f"    ENTSO-E error [{start.date()} → {end.date()}]: {exc}")
        return pd.DataFrame(columns=["price_eur_mwh"])


def find_missing_ranges(df: pd.DataFrame, full_index: pd.DatetimeIndex):
    """
    Zwraca listę (start, end) przedziałów, w których brakuje cen.
    Uwzględnia zarówno NaN w istniejących wierszach, jak i całkowity brak wierszy.
    """
    # Reindeksuj na pełny godzinowy szkielet
    df_full = df.reindex(full_index)
    missing = df_full[df_full["price_eur_mwh"].isna()].index

    if len(missing) == 0:
        return []

    # Grupuj ciągłe luki w przedziały (unikamy setek osobnych zapytań do API)
    ranges = []
    gap_start = missing[0]
    prev = missing[0]

    for ts in missing[1:]:
        if ts - prev > pd.Timedelta(hours=1):
            ranges.append((gap_start, prev + pd.Timedelta(hours=1)))
            gap_start = ts
        prev = ts

    ranges.append((gap_start, prev + pd.Timedelta(hours=1)))
    return ranges


def fill_missing_prices():
    if not ENTSOE_API_KEY:
        print("Brak klucza ENTSOE_API_KEY w .env – przerwano.")
        return

    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
    end_target = get_end_target()

    # ── 1. Wczytaj istniejący plik lub stwórz pusty ──────────────────────────
    if PRICES_FILE.exists():
        df = pd.read_parquet(PRICES_FILE)

        # Ujednolić indeks → UTC
        if "date" in df.columns:
            df = df.set_index("date")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "date"
        df = df.sort_index()

        print(f"Wczytano {len(df)} rekordów: {df.index.min().date()} → {df.index.max().date()}")
    else:
        print(f"Plik {PRICES_FILE} nie istnieje – tworzę nowy od {START_DATE.date()}.")
        df = pd.DataFrame(columns=["price_eur_mwh"])
        df.index.name = "date"

    # ── 2. Pełny godzinowy szkielet ──────────────────────────────────────────
    full_index = pd.date_range(start=START_DATE, end=end_target, freq="1h", tz="UTC")

    # ── 3. Wykryj luki ───────────────────────────────────────────────────────
    gaps = find_missing_ranges(df[["price_eur_mwh"]] if "price_eur_mwh" in df.columns
                               else pd.DataFrame(index=df.index, columns=["price_eur_mwh"]),
                               full_index)

    if not gaps:
        print("Brak luk – plik jest kompletny.")
        return

    total_missing = sum(int((e - s).total_seconds() // 3600) for s, e in gaps)
    print(f"\nWykryto {len(gaps)} przedziałów z brakami ({total_missing} godzin łącznie):")
    for s, e in gaps:
        print(f"  {s}  →  {e}")

    # ── 4. Pobierz brakujące dane ─────────────────────────────────────────────
    fetched_frames = []
    for i, (gap_s, gap_e) in enumerate(gaps, 1):
        print(f"\n[{i}/{len(gaps)}] Pobieranie {gap_s.date()} → {gap_e.date()} ...", end=" ", flush=True)
        chunk = fetch_prices_range(client, gap_s, gap_e)
        if not chunk.empty:
            print(f"OK ({len(chunk)} rekordów)")
            fetched_frames.append(chunk)
        else:
            print("brak danych (jeszcze nie opublikowane lub błąd API)")

    if not fetched_frames:
        print("\nENTSO-E nie zwróciło żadnych nowych danych.")
        return

    df_new = pd.concat(fetched_frames)
    df_new = df_new[~df_new.index.duplicated(keep="last")]

    # ── 5. Scal z istniejącym plikiem ─────────────────────────────────────────
    df_combined = pd.concat([df, df_new])
    df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()

    # Zachowaj tylko kolumnę price_eur_mwh (reszta pochodzi z merge_data)
    if "price_eur_mwh" in df_combined.columns:
        df_combined = df_combined[["price_eur_mwh"]]

    DATA_DIR.mkdir(exist_ok=True)
    df_combined.to_parquet(PRICES_FILE)

    filled = len(df_new)
    print(f"\nGotowe. Uzupełniono {filled} godzin.")
    print(f"Nowy zakres pliku: {df_combined.index.min().date()} → {df_combined.index.max().date()}")
    print(f"Rekordów łącznie : {len(df_combined)}")

    # ── 6. Sprawdzenie pozostałych luk ────────────────────────────────────────
    remaining = find_missing_ranges(df_combined[["price_eur_mwh"]], full_index)
    now_utc = pd.Timestamp.now(tz="UTC")
    # Filtruj luki które są w przyszłości (ENTSO-E jeszcze nie opublikowało)
    remaining_past = [(s, e) for s, e in remaining if s < now_utc]
    if remaining_past:
        print(f"\nPozostałe luki w danych historycznych ({len(remaining_past)} przedziałów):")
        for s, e in remaining_past:
            print(f"  {s} → {e}")
        print("(Mogą wynikać z braku danych po stronie ENTSO-E dla tych godzin.)")
    else:
        print("\nWszystkie historyczne godziny są uzupełnione.")


if __name__ == "__main__":
    fill_missing_prices()