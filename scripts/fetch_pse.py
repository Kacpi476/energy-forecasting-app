# fetch_pse.py
import requests
import pandas as pd
import ast
from datetime import timedelta

BASE_URL = (
    "https://api.raporty.pse.pl/api/his-wlk-cal"
    "?$filter=business_date eq '{date}'"
    "&$orderby=dtime_utc asc"
    "&$first=20000"
)

TARGET_COLUMNS = ["dtime_utc", "demand", "pv", "wi", "jg", "jnwrb"]

def fetch_pse(start, end):
    """
    Pobiera dane PSE (demand + produkcję) dla zakresu start -> end.
    Zwraca DataFrame z indeksem 'date' UTC i wszystkimi kolumnami.
    Obsługuje już istniejące dane – pobiera tylko brakujące dni.
    """
    all_days = []

    current = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        url = BASE_URL.format(date=date_str)
        print(f"PSE data: {date_str}")

        try:
            resp = requests.get(url)
            if resp.status_code != 200:
                print(f"Brak danych dla {date_str}, status: {resp.status_code}")
                current += timedelta(days=1)
                continue

            data = resp.json()
            if not data:
                current += timedelta(days=1)
                continue

            df = pd.DataFrame(data)

            for col in df.select_dtypes(include="object"):
                df[col] = df[col].map(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else x
                )

            rows = []
            for _, row in df.iterrows():
                cell = row.iloc[0]
                if isinstance(cell, dict):
                    rows.append(cell)
                else:
                    rows.append(row)
            df2 = pd.DataFrame(rows)

            if not set(TARGET_COLUMNS).issubset(df2.columns):
                current += timedelta(days=1)
                continue

            df2 = df2[TARGET_COLUMNS].copy()
            df2["dtime_utc"] = pd.to_datetime(df2["dtime_utc"], utc=True)
            df2 = df2.sort_values("dtime_utc")

            all_days.append(df2)

        except Exception as e:
            print(f"⚠ Błąd pobierania {date_str}: {e}")

        current += timedelta(days=1)

    if not all_days:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    df = pd.concat(all_days).dropna()

    df = df.set_index("dtime_utc").sort_index()
    df.index.name = "date"

    return df
