import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def get_latest_forecast():
    """
    Generuje prognozę dla najnowszego dostępnego rekordu w zbiorze danych.
    Używana przez app.py do wyświetlenia aktualnej predykcji w dashboardzie.

    Returns:
        tuple: (predicted_price: float, timestamp: pd.Timestamp)
    """
    DATA_PATH = Path("data/final_training_data.parquet")
    MODEL_PATH = Path("models/price_rf_model.pkl")
    FEATURES_PATH = Path("models/feature_names.pkl")

    df = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # --- Odtworzenie cech pochodnych (muszą być identyczne jak przy trenowaniu) ---
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["res_share"] = (df["pv"] + df["wi"]) / df["demand"].replace(0, 1)

    # POPRAWKA: price_lag_24 zamiast błędnego price_lag_1
    # Model trenowany był na cesze price_lag_24 (cena sprzed 24h)
    df["price_lag_24"] = df["price_eur_mwh"].shift(24)

    # --- Wybór wiersza do predykcji ---
    # Bierzemy ostatni rekord; jeśli ma braki — cofamy się o jeden
    latest_row = df.tail(1).copy()

    if latest_row[features].isnull().values.any():
        print("⚠ Ostatni rekord niekompletny — używam poprzedniego.")
        latest_row = df.tail(2).head(1).copy()

    prediction = model.predict(latest_row[features])[0]
    timestamp = latest_row["date"].iloc[0]

    return prediction, timestamp


if __name__ == "__main__":
    price, ts = get_latest_forecast()
    print(f"Prognoza dla {ts}: {price:.2f} EUR/MWh")