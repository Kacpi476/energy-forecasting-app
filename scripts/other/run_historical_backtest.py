"""
run_historical_backtest.py
--------------------------
Przelicza predicted_price dla całego okresu historycznego (walk-forward).
Dla każdego dnia T generuje prognozę używając wyłącznie danych dostępnych
przed T (brak data leakage). Wynik łączy z bieżącymi prognozami operacyjnymi
z forecast_history.parquet — nie nadpisuje przyszłości.

Uruchom raz przed obroną żeby sekcja 3 (MAE tygodniowy) miała pełne dane.
Czas działania: ~2-5 minut.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

DATA_DIR   = Path("data")
MODEL_PATH = Path("models/price_rf_model.pkl")
FEATURES_PATH = Path("models/feature_names.pkl")

# Backtest od kiedy — pierwsze pełne dane treningowe
BACKTEST_START = pd.Timestamp("2025-01-01", tz="UTC")


def run_historical_backtest():
    if not MODEL_PATH.exists():
        print("Brak modelu. Uruchom najpierw train_model_final.py.")
        return

    df = pd.read_parquet(DATA_DIR / "final_training_data.parquet")
    model     = joblib.load(MODEL_PATH)
    features  = joblib.load(FEATURES_PATH)

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    df["day"]  = df["date"].dt.date

    # Kolumna wynikowa
    df["predicted_price"] = np.nan

    # Dni z kompletem 24h cen — na nich będziemy bazować lagi
    price_counts = df[df["price_eur_mwh"].notna()].groupby("day").size()
    full_days    = sorted(price_counts[price_counts == 24].index.tolist())

    # Zakres backtestowy: od BACKTEST_START do przedostatniego pełnego dnia
    # (ostatni zostawiamy dla bieżącej prognozy)
    backtest_days = [d for d in full_days if pd.Timestamp(d, tz="UTC") >= BACKTEST_START]

    if len(backtest_days) < 2:
        print("Za mało danych historycznych do backtestowania.")
        return

    print(f"Backtest: {backtest_days[0]} → {backtest_days[-1]} ({len(backtest_days)} dni)")

    # Walk-forward: dla każdego dnia T generujemy prognozę
    # używając price_lag_24 z dnia T-1 (realne ceny, bez leakage)
    for target_day in tqdm(backtest_days, desc="Backtest"):
        day_mask = df["day"] == target_day
        rows     = df[day_mask]

        if len(rows) < 24:
            continue

        # price_lag_24: shift(24) na całym df do tego dnia włącznie
        df_to_day = df[df["day"] <= target_day].copy()
        df_to_day["price_lag_24"] = df_to_day["price_eur_mwh"].shift(24).ffill()

        X = df_to_day.loc[day_mask, features].copy()

        if X.isna().any().any():
            X = X.ffill().bfill()

        preds = model.predict(X)
        df.loc[day_mask, "predicted_price"] = preds

    # --- Połącz z bieżącymi prognozami operacyjnymi ---
    forecast_path = DATA_DIR / "forecast_history.parquet"
    if forecast_path.exists():
        current = pd.read_parquet(forecast_path)
        current["date"] = pd.to_datetime(current["date"], utc=True)

        # Zachowaj bieżące prognozy operacyjne (przyszłość)
        last_backtest_day = pd.Timestamp(backtest_days[-1], tz="UTC")
        future_preds = current[
            current["date"] > last_backtest_day + pd.Timedelta(hours=23)
        ][["date", "predicted_price"]].copy()

        if not future_preds.empty:
            # Wpisz przyszłe prognozy do głównego df
            df = df.merge(
                future_preds.rename(columns={"predicted_price": "pred_future"}),
                on="date", how="left"
            )
            future_mask = df["pred_future"].notna()
            df.loc[future_mask, "predicted_price"] = df.loc[future_mask, "pred_future"]
            df.drop(columns=["pred_future"], inplace=True)
            print(f"Zachowano {len(future_preds)} godzin bieżącej prognozy operacyjnej.")

    # Zapis
    output = df[["date", "price_eur_mwh", "predicted_price"]].copy()
    output.to_parquet(forecast_path, index=False)

    overlap = (output["predicted_price"].notna() & output["price_eur_mwh"].notna()).sum()
    mae = (output["price_eur_mwh"] - output["predicted_price"]).abs().mean()

    print(f"\nGotowe!")
    print(f"  Predicted notna : {output['predicted_price'].notna().sum()}")
    print(f"  Price notna     : {output['price_eur_mwh'].notna().sum()}")
    print(f"  Overlap (MAE)   : {overlap} godzin")
    print(f"  MAE globalny    : {mae:.2f} EUR/MWh")


if __name__ == "__main__":
    run_historical_backtest()