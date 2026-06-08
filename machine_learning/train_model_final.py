import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Konfiguracja ---
START_DATE = "2024-07-01"
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def merge_and_clean_data() -> pd.DataFrame:
    """
    Wczytuje surowe pliki Parquet, wykonuje resampling do 1h,
    łączy wszystkie źródła i generuje cechy pochodne.
    """
    print("Wczytywanie i przygotowanie danych...")

    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    pse = pd.read_parquet(DATA_DIR / "pse.parquet")
    weather = pd.read_parquet(DATA_DIR / "weather.parquet")
    co2 = pd.read_parquet(DATA_DIR / "co2.parquet")

    # Resampling do interwału godzinowego
    prices_h = prices.resample("1h").mean()
    pse_h = pse.select_dtypes(include=["number"]).resample("1h").mean()
    weather_h = weather.select_dtypes(include=["number"]).resample("1h").mean()

    # Outer join — zachowujemy przyszłość z danych pogodowych
    df = prices_h.join([pse_h, weather_h], how="outer").sort_index()

    # CO2: merge_asof z direction='backward' — ostatnia znana dzienna cena
    df = pd.merge_asof(
        df, co2.sort_index(),
        left_index=True, right_index=True,
        direction="backward"
    )

    df = df.reset_index()
    df.rename(columns={df.columns[0]: "date"}, inplace=True)

    # POPRAWKA: tz_localize zamiast błędnego dt.localize
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")
    else:
        df["date"] = df["date"].dt.tz_convert("UTC")

    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek

    # Profil popytu: typowy popyt godzinowy na podstawie historii
    last_real_ts = prices_h.index.max()
    if last_real_ts.tzinfo is None:
        last_real_ts = last_real_ts.tz_localize("UTC")

    history_mask = df["date"] <= last_real_ts
    future_mask = df["date"] > last_real_ts

    demand_profile = (
        df[history_mask]
        .groupby(["day_of_week", "hour"])["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "typical_demand"})
    )
    df = df.merge(demand_profile, on=["day_of_week", "hour"], how="left")
    df.loc[future_mask, "demand"] = df.loc[future_mask, "typical_demand"]
    df.drop(columns=["typical_demand"], inplace=True)

    # Imputacja: ffill/bfill dla zmiennych ciągłych
    cols_to_fill = ["co2_price_eur", "temperature_c", "wind_speed_ms", "solar_wm2", "pv", "wi"]
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill()

    # Korekta PV — zerowanie w godzinach nocnych (brak produkcji słonecznej)
    df.loc[(df["hour"] < 6) | (df["hour"] > 20), "pv"] = 0

    # Cechy pochodne
    df["price_lag_24"] = df["price_eur_mwh"].shift(24).ffill()
    df["res_share"] = (df["pv"] + df["wi"]) / df["demand"].replace(0, 1)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Filtracja zakresu i usuwanie braków w kluczowych cechach
    df = df[df["date"] >= pd.Timestamp(START_DATE, tz="UTC")]
    df = df.dropna(subset=["demand", "co2_price_eur", "price_lag_24"])

    return df


def train_test_split_chronological(X, y, test_size: float = 0.15):
    """
    Podział chronologiczny zbioru danych (bez losowego mieszania).
    Zapobiega zjawisku look-ahead bias w modelowaniu szeregów czasowych.
    """
    split_idx = int(len(X) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def train_model(df: pd.DataFrame):
    """
    Trenuje model Random Forest Regressor na przygotowanym zbiorze danych.
    Zapisuje model i listę cech do katalogu models/.
    """
    print("Trenowanie modelu Random Forest...")

    features = [
        "demand", "pv", "wi", "co2_price_eur",
        "temperature_c", "wind_speed_ms", "solar_wm2",
        "hour_sin", "hour_cos", "day_of_week",
        "res_share", "price_lag_24"
    ]
    target = "price_eur_mwh"

    # Trenujemy wyłącznie na rekordach z rzeczywistą ceną (nie na prognozach)
    train_df = df.dropna(subset=[target]).copy()

    X = train_df[features]
    y = train_df[target]

    # Chronologiczny podział: 85% trening, 15% test (najnowsze dane)
    X_train, X_test, y_train, y_test = train_test_split_chronological(X, y, test_size=0.15)

    print(f"Zbiór treningowy: {len(X_train)} rekordów | Testowy: {len(X_test)} rekordów")

    # Model Random Forest — parametry zoptymalizowane pod rynek energii
    # n_estimators=200, max_depth=12, min_samples_leaf=5
    # zapewniają balans między złożonością a generalizacją
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Ewaluacja na zbiorze testowym (dane chronologicznie najnowsze)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print("-" * 40)
    print(f"Wyniki modelu (zbiór testowy):")
    print(f"  R²  : {r2:.4f}  (model wyjaśnia {r2 * 100:.1f}% wariancji cen)")
    print(f"  MAE : {mae:.2f} EUR/MWh")
    print("-" * 40)

    # Serializacja modelu i listy cech (joblib — format binarny .pkl)
    joblib.dump(model, MODELS_DIR / "price_rf_model.pkl")
    joblib.dump(features, MODELS_DIR / "feature_names.pkl")

    # Zapis zbioru treningowego z cechami pochodnymi
    df.to_parquet("data/final_training_data.parquet", index=False)

    print("Model i dane zapisane.")
    print(f"  Plik modelu : models/price_rf_model.pkl")
    print(f"  Dane        : data/final_training_data.parquet")


if __name__ == "__main__":
    merged_df = merge_and_clean_data()
    train_model(merged_df)