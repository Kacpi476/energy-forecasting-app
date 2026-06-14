import pandas as pd
import numpy as np
from pathlib import Path

START_DATE = "2024-07-01"


def merge_datasets():
    DATA_DIR = Path(__file__).parent.parent / "data"

    prices  = pd.read_parquet(DATA_DIR / "prices.parquet")
    pse     = pd.read_parquet(DATA_DIR / "pse.parquet")
    weather = pd.read_parquet(DATA_DIR / "weather.parquet")
    co2     = pd.read_parquet(DATA_DIR / "co2.parquet")

    # --- Resampling do interwału godzinowego ---
    prices_h  = prices.resample("1h").mean()
    pse_h     = pse.select_dtypes(include=["number"]).resample("1h").mean()
    weather_h = weather.select_dtypes(include=["number"]).resample("1h").mean()

    # --- Szkielet oparty na weather (najdłuższy horyzont: +72h) ---
    # PSE ma krótszy zakres (dane operacyjne dostępne z opóźnieniem),
    # dlatego joinujemy na indeksie weather i PSE uzupełniamy ffill.
    # Dzięki temu nie ma dziur w final_training_data dla godzin bez PSE.
    df = weather_h.join(prices_h, how="left").join(pse_h, how="left").sort_index()

    # PSE: uzupełnij brakujące godziny (po końcu danych PSE) profilem historycznym
    # — ffill tymczasowo, potem zastąpimy demand profilem dobowym
    pse_cols = ["demand", "pv", "wi", "jg", "jnwrb"]
    existing_pse_cols = [c for c in pse_cols if c in df.columns]
    df[existing_pse_cols] = df[existing_pse_cols].ffill()

    # --- CO2: merge_asof z direction='backward' ---
    df = pd.merge_asof(
        df, co2.sort_index(),
        left_index=True, right_index=True,
        direction="backward"
    )

    df = df.reset_index()
    df.rename(columns={df.columns[0]: "date"}, inplace=True)

    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")
    else:
        df["date"] = df["date"].dt.tz_convert("UTC")

    df["hour"]        = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek

    # --- Profil popytu: zastępuje ffill-owane demand w strefie bez PSE ---
    # Używamy tylko rekordów z ceną (historia) do budowy profilu
    history_mask = df["price_eur_mwh"].notna()
    demand_profile = (
        df[history_mask]
        .groupby(["day_of_week", "hour"])["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "typical_demand"})
    )
    df = df.merge(demand_profile, on=["day_of_week", "hour"], how="left")

    # Wyznacz strefę gdzie PSE nie ma już danych (po max indeksie PSE)
    pse_max_ts = pse_h.index.max()
    if pse_max_ts.tzinfo is None:
        pse_max_ts = pse_max_ts.tz_localize("UTC")
    no_pse_mask = df["date"] > pse_max_ts

    # W strefie bez PSE: demand z profilu historycznego, pv/wi z ffill (już zrobione)
    df.loc[no_pse_mask, "demand"] = df.loc[no_pse_mask, "typical_demand"]
    df.drop(columns=["typical_demand"], inplace=True)

    # --- Imputacja pozostałych zmiennych ---
    cols_to_fill = ["co2_price_eur", "temperature_c", "wind_speed_ms", "solar_wm2", "pv", "wi"]
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill()

    # --- Korekta PV: brak produkcji słonecznej w nocy ---
    df.loc[(df["hour"] < 6) | (df["hour"] > 19), "pv"] = 0

    # --- Cechy pochodne ---
    df["price_lag_24"] = df["price_eur_mwh"].shift(24)
    df["res_share"]    = (df["pv"] + df["wi"]) / df["demand"].replace(0, 1)
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)

    # --- Filtracja zakresu ---
    df = df[df["date"] >= pd.Timestamp(START_DATE, tz="UTC")]
    # Usuwamy tylko rekordy bez pogody (temperature_c) — PSE i CO2 są już uzupełnione
    df = df.dropna(subset=["temperature_c", "co2_price_eur"])

    return df


if __name__ == "__main__":
    df_result = merge_datasets()
    df_result.to_parquet("data/final_training_data.parquet", index=False)

    last_price   = df_result[df_result["price_eur_mwh"].notna()]["date"].max()
    future_count = df_result[df_result["price_eur_mwh"].isna()].shape[0]
    print(f"Dane zmergowane. Zakres: {df_result['date'].min().date()} -> {df_result['date'].max().date()}")
    print(f"Ostatnia cena rzeczywista : {last_price}")
    print(f"Godzin do prognozowania   : {future_count}")
    print(f"Ciaglosc rekordow         : {len(df_result)} (oczekiwane ~{future_count + int(last_price.timestamp()//3600 - pd.Timestamp(START_DATE, tz='UTC').timestamp()//3600)})")