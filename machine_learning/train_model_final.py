import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

def train_energy_model():
    DATA_PATH = Path("data/final_training_data.parquet")
    MODEL_DIR = Path("machine_learning")
    MODEL_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        print("Błąd: Brak pliku final_training_data.parquet. Uruchom najpierw update_data.py")
        return

    df = pd.read_parquet(DATA_PATH)
    
    df = df.dropna(subset=['price_eur_mwh'])

    features = [
    'demand', 'pv', 'wi', 'co2_price_eur', 
    'temperature_c', 'wind_speed_ms', 'solar_wm2',
    'hour_sin', 'hour_cos', 'day_of_week', 
    'res_share', 'price_lag_24'
    ]
    target = 'price_eur_mwh'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"Trenowanie modelu na {len(X_train)} rekordach...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model wytrenowany!")
    print(f"Wyniki na zbiorze testowym:")
    print(f"   - MAE (Średni błąd): {mae:.2f} EUR/MWh")
    print(f"   - R2 Score: {r2:.4f}")

    joblib.dump(model, MODEL_DIR / "model.joblib")
    print(f"Model zapisany w: {MODEL_DIR / 'model.joblib'}")

if __name__ == "__main__":
    train_energy_model()