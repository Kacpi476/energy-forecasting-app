import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
from xgboost import XGBRegressor

def train_price_model():
    # 1. Wczytanie danych (obsługa CSV z Twojego pliku)
    data_path = Path("data/final_training_data.parquet")
    if not data_path.exists():
        print("❌ Błąd: Brak pliku final_training_data.parquet!")
        return

    df = pd.read_parquet(data_path)
    
    # 2. Definicja cech (Features)
    features = [
        'demand', 'pv', 'wi', 'co2_price_eur', 
        'temperature_c', 'wind_speed_ms', 'solar_wm2',
        'hour_sin', 'hour_cos', 'day_of_week', 
        'res_share', 'price_lag_24'
    ]
    target = 'price_eur_mwh'

    # 3. Czyszczenie danych
    # Usuwamy wiersze, gdzie cena jest pusta (te, które model ma dopiero prognozować)
    train_df = df.dropna(subset=[target] + features)

    X = train_df[features]
    y = train_df[target]

    # 4. Podział chronologiczny (nie mieszamy losowo, bo to szereg czasowy!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    print(f"🚀 Trenowanie na {len(X_train)} rekordach (Zakres do końca lutego)...")

    # 5. NOWE PARAMETRY - Klucz do "ożywienia" modelu
    model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
    )   
    
    model.fit(X_train, y_train)

    # 6. Ewaluacja
    preds = model.predict(X_test)
    print(f"\n📊 Wyniki modelu po aktualizacji danych:")
    print(f"   - Błąd MAE: {mean_absolute_error(y_test, preds):.2f} EUR")
    print(f"   - Dopasowanie R2: {r2_score(y_test, preds):.4f}")

    # 7. Zapis
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, 'models/price_rf_model.pkl')
    joblib.dump(features, 'models/feature_names.pkl')
    
    print("\n✅ Model nauczony na Twoich danych i zapisany!")

if __name__ == "__main__":
    train_price_model()