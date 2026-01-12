import pandas as pd
import joblib
from pathlib import Path

def get_latest_forecast():
    # 1. Ścieżki
    DATA_PATH = Path("data/final_training_data.parquet")
    MODEL_PATH = Path("models/price_rf_model.pkl")
    FEATURES_PATH = Path("models/feature_names.pkl")

    # 2. Wczytanie danych i modelu
    df = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    
    # 3. WYMAGANE: Wyliczenie brakujących kolumn (Feature Engineering)
    # Musimy powtórzyć te same operacje co w train_model.py
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['price_lag_1'] = df['price_eur_mwh'].shift(1)
    df['res_share'] = (df['pv'] + df['wi']) / df['demand']
    
    # Bierzemy ostatni wiersz (najświeższe dane)
    latest_row = df.tail(1)
    
    # Sprawdzenie czy nie mamy NaN w ostatnim wierszu (np. przez shift)
    if latest_row[features].isnull().values.any():
        # Jeśli ostatni jest NaN, weź przedostatni (do testów)
        latest_row = df.tail(2).head(1)

    # 4. Predykcja
    prediction = model.predict(latest_row[features])[0]
    
    return prediction, latest_row['date'].iloc[0]