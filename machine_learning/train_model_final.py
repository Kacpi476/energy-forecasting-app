import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# KONFIGURACJA
START_DATE = "2024-07-01"
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def merge_and_clean_data():
    print("⏳ Wczytywanie i czyszczenie danych...")
    
    # 1. Wczytanie surowych plików parquet
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    pse = pd.read_parquet(DATA_DIR / "pse.parquet")
    weather = pd.read_parquet(DATA_DIR / "weather.parquet")
    co2 = pd.read_parquet(DATA_DIR / "co2.parquet")

    # 2. Agregacja godzinowa
    prices_h = prices.resample('1h').mean()
    pse_h = pse.select_dtypes(include=['number']).resample('1h').mean()
    weather_h = weather.select_dtypes(include=['number']).resample('1h').mean()

    # 3. Łączenie (Outer join, żeby nie zgubić przyszłości z pogody)
    df = prices_h.join([pse_h, weather_h], how='outer').sort_index()

    # 4. CO2 - ceny rynkowe
    df = pd.merge_asof(
        df, co2.sort_index(), 
        left_index=True, right_index=True, direction='backward'
    )

    df = df.reset_index()
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    
    # Obsługa stref czasowych
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.localize('UTC')
    else:
        df['date'] = df['date'].dt.tz_convert('UTC')

    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek

    # --- INTELIGENTNA NAPRAWA PRZYSZŁOŚCI (ZAMIAST FFILL) ---
    # Ustalamy moment, gdzie kończą się realne dane rynkowe
    last_real_ts = prices_h.index.max()
    if last_real_ts.tzinfo is None:
        last_real_ts = last_real_ts.tz_localize('UTC')

    history_mask = df['date'] <= last_real_ts
    future_mask = df['date'] > last_real_ts

    # Tworzymy wzorzec popytu na podstawie historii (Godzina + Dzień tygodnia)
    # To sprawi, że popyt w przyszłości będzie "falował"
    demand_profile = df[history_mask].groupby(['day_of_week', 'hour'])['demand'].mean().reset_index()
    demand_profile.rename(columns={'demand': 'typical_demand'}, inplace=True)
    
    df = df.merge(demand_profile, on=['day_of_week', 'hour'], how='left')
    df.loc[future_mask, 'demand'] = df.loc[future_mask, 'typical_demand']
    df.drop(columns=['typical_demand'], inplace=True)

    # Pogodę i CO2 możemy dociągnąć ffill-em (jeśli nie ma prognozy)
    cols_to_fill = ['co2_price_eur', 'temperature_c', 'wind_speed_ms', 'solar_wm2', 'pv', 'wi']
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill()

    # Korekta PV - słońce nie świeci w nocy
    df.loc[(df['hour'] < 6) | (df['hour'] > 20), 'pv'] = 0

    # 5. Obliczanie cech końcowych
    df['price_lag_24'] = df['price_eur_mwh'].shift(24).ffill()
    df['res_share'] = (df['pv'] + df['wi']) / df['demand'].replace(0, 1)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Filtracja zakresu i usuwanie braków w cechach
    df = df[df['date'] >= pd.Timestamp(START_DATE, tz='UTC')]
    df = df.dropna(subset=['demand', 'co2_price_eur', 'price_lag_24'])
    
    return df

def train_model(df):
    print("🚀 Trenowanie modelu...")
    
    # Wybieramy cechy
    features = [
        'demand', 'pv', 'wi', 'co2_price_eur', 
        'temperature_c', 'wind_speed_ms', 'solar_wm2',
        'hour_sin', 'hour_cos', 'day_of_week', 
        'res_share', 'price_lag_24'
    ]
    target = 'price_eur_mwh'

    # Trening TYLKO na danych z ceną
    train_df = df.dropna(subset=[target]).copy()
    
    X = train_df[features]
    y = train_df[target]

    # Podział chronologiczny
    X_train, X_test, y_train, y_test = train_test_split_chronological(X, y, test_size=0.15)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # Ewaluacja
    preds = model.predict(X_test)
    print(f"✅ Model gotowy. R2: {r2_score(y_test, preds):.4f}, MAE: {mean_absolute_error(y_test, preds):.2f} EUR")

    # Zapis
    joblib.dump(model, MODELS_DIR / 'price_rf_model.pkl')
    joblib.dump(features, MODELS_DIR / 'feature_names.pkl')
    df.to_parquet("data/final_training_data.parquet", index=False)
    print("💾 Model i dane zapisane do final_training_data.parquet")

def train_test_split_chronological(X, y, test_size=0.15):
    split_idx = int(len(X) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

if __name__ == "__main__":
    merged_df = merge_and_clean_data()
    train_model(merged_df)