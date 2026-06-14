import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def run_backtest():
    """
    Silnik prognozowania Day-Ahead:
    - Identyfikuje ostatni pełny dzień z cenami rzeczywistymi (T)
    - Generuje prognozę dla WSZYSTKICH godzin po T (dzień bieżący + kolejne doby)
    - Dopisuje nowe prognozy do forecast_history.parquet BEZ nadpisywania historii

    UWAGA: forecast_mask nie filtruje po isna() — prognozujemy cały zakres od T+1,
    niezależnie od tego czy godziny bieżącego dnia mają już ceny częściowe.
    Dzięki temu app.py zawsze ma ciągłą linię prognozy bez przerw.
    """
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_PATH = BASE_DIR / "models/price_rf_model.pkl"
    FEATURES_PATH = BASE_DIR / "models/feature_names.pkl"

    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        print("Błąd: Brak pliku modelu. Uruchom najpierw train_model_final.py.")
        return

    df = pd.read_parquet(DATA_DIR / "final_training_data.parquet")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    df["day"] = df["date"].dt.date

    # --- Znajdź ostatni dzień z kompletem 24h cen rzeczywistych ---
    price_counts = df[df["price_eur_mwh"].notna()].groupby("day").size()
    full_days = price_counts[price_counts == 24].index.tolist()

    if not full_days:
        print("Błąd: Brak pełnych dób (24h) z cenami w pliku!")
        return

    last_full_day = max(full_days)
    print(f"Ostatni pełny dzień z cenami: {last_full_day}")

    # --- Wyznacz zakres prognozy: WSZYSTKIE rekordy po ostatnim pełnym dniu ---
    forecast_mask = df["day"] > last_full_day
    forecast_rows = df[forecast_mask]

    if forecast_rows.empty:
        print("Brak godzin do prognozowania — dane są aktualne.")
        return

    forecast_days = sorted(forecast_rows["day"].unique())
    print(f"Generuję prognozę dla {len(forecast_rows)}h "
          f"({forecast_days[0]} -> {forecast_days[-1]})")

    # --- Bezpieczne obliczenie price_lag_24 ---
    plot_df = df.copy()
    plot_df["price_lag_24"] = plot_df["price_eur_mwh"].shift(24).ffill()

    lag_missing = plot_df.loc[forecast_mask, "price_lag_24"].isna().sum()
    if lag_missing > 0:
        print(f"Brakuje {lag_missing} wartosci price_lag_24 — uzupelniam bfill.")
        plot_df["price_lag_24"] = plot_df["price_lag_24"].bfill()

    # --- Predykcja ---
    X = plot_df.loc[forecast_mask, features].copy()

    missing_cols = X.columns[X.isna().any()].tolist()
    if missing_cols:
        print(f"Braki danych w cechach: {missing_cols}. Uzupelniam ffill/bfill.")
        X = X.ffill().bfill()

    day_preds = model.predict(X)
    plot_df.loc[forecast_mask, "predicted_price"] = day_preds

    # --- Zapis wyników: dopisuj do istniejącego pliku, nie nadpisuj ---
    forecast_path = DATA_DIR / "forecast_history.parquet"
    new_forecasts = plot_df.loc[forecast_mask, ["date", "price_eur_mwh", "predicted_price"]].copy()

    if forecast_path.exists():
        old = pd.read_parquet(forecast_path)
        old["date"] = pd.to_datetime(old["date"], utc=True)
        # Zachowaj historyczny backtest, dopisz tylko nowe prognozy operacyjne
        combined = pd.concat([old, new_forecasts])
        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    else:
        combined = new_forecasts

    combined.to_parquet(forecast_path, index=False)

    print(f"Prognoza wygenerowana.")
    print(f"   Godzin: {len(day_preds)} | "
          f"Srednia: {day_preds.mean():.2f} | Min: {day_preds.min():.2f} | Max: {day_preds.max():.2f} EUR/MWh")
    print(f"   Zakres forecast_history: {combined['date'].min().date()} -> {combined['date'].max().date()}")
    print(f"   Ostatnia cena realna: {combined[combined['price_eur_mwh'].notna()]['date'].max()}")
    print(f"   Pierwsza prognoza:    {combined[combined['predicted_price'].notna()]['date'].min()}")


if __name__ == "__main__":
    run_backtest()