import requests
import pandas as pd
import ast
from datetime import datetime, timedelta

BASE_URL = (
    "https://api.raporty.pse.pl/api/his-wlk-cal"
    "?$filter=business_date eq '{date}'"
    "&$orderby=dtime_utc asc"
    "&$first=20000"
)

START_DATE = datetime(2024, 7, 1)
END_DATE = datetime.now()

TARGET_COLUMNS = ["dtime_utc", "demand", "pv", "wi", "jg", "jnwrb"]

OUTPUT_FILE = "data/pse_full_history_clean.csv"

def pobierz_i_wyczysc_dzien(date_str):
    url = BASE_URL.format(date=date_str)
    print(f"🔄 Pobieram: {date_str}")

    resp = requests.get(url)

    if resp.status_code != 200:
        print(f"   ❌ Błąd HTTP {resp.status_code}")
        return None

    data = resp.json()
    if not data:
        print("   ⚠️ Brak danych tego dnia")
        return None

    df = pd.DataFrame(data)

    df = df.applymap(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else x)

    rows = []
    for _, row in df.iterrows():
        if isinstance(row[0], dict):
            rows.append(row[0])
        else:
            rows.append(row)

    df2 = pd.DataFrame(rows)

    missing = [c for c in TARGET_COLUMNS if c not in df2.columns]
    if missing:
        print(f"Brak kolumn: {missing}")
        return None

    df2 = df2[TARGET_COLUMNS].copy()

    df2["dtime_utc"] = pd.to_datetime(df2["dtime_utc"])
    df2 = df2.sort_values("dtime_utc")

    print(f"   ✔ OK — {len(df2)} rekordów")
    return df2



all_days = []
current = START_DATE

while current <= END_DATE:
    ds = current.strftime("%Y-%m-%d")
    df_day = pobierz_i_wyczysc_dzien(ds)

    if df_day is not None:
        all_days.append(df_day)

    current += timedelta(days=1)

if not all_days:
    print("Nie pobrano żadnych danych.")
else:
    final_df = pd.concat(all_days, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nZapisano pełny plik: {OUTPUT_FILE}")
    print(f"Łącznie rekordów: {len(final_df)}")
