# fetch_prices.py
from entsoe import EntsoePandasClient
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

def fetch_prices(start, end):
    """
    Pobiera ceny day-ahead z ENTSOE.
    Obsługuje puste zakresy i konwertuje indeks na UTC.
    """
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

    start = pd.Timestamp(start).tz_convert("Europe/Brussels")
    end = pd.Timestamp(end).tz_convert("Europe/Brussels")

    # Jeśli start >= end, zwracamy pusty DF
    if start >= end:
        return pd.DataFrame(columns=["price_eur_mwh"])

    prices = client.query_day_ahead_prices(
        country_code="PL",
        start=start,
        end=end
    )

    df = prices.to_frame(name="price_eur_mwh")

    df.index = df.index.tz_convert("UTC")

    if "price" in df.columns:
        df = df.drop(columns=["price"], errors="ignore")

    df["date"] = df.index
    df = df.set_index("date").sort_index()

    return df
