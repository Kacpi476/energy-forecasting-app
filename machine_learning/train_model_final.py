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
        print("❌ Błąd: Brak pliku final_training_data.parquet. Uruchom najpierw update_data.py")
        return

    # 1. Wczytanie danych
    df = pd.read_parquet(DATA_PATH)
    
    # Usuwamy wiersze, gdzie nie ma ceny rzeczywistej (nie możemy na nich trenować)
    df = df.dropna(subset=['price_eur_mwh'])

    # 2. Definicja cech (Features) i celu (Target)
    # Muszą być identyczne z tymi, które generujemy w merge_data.py
    features = ['demand', 'pv', 'wi', 'co2_price_eur', 'hour', 'day_of_week', 'res_share', 'price_lag_1']
    target = 'price_eur_mwh'

    X = df[features]
    y = df[target]

    # 3. Podział na zbiór treningowy i testowy (chronologiczny!)
    # W szeregach czasowych nie robimy losowego splitu (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📊 Trenowanie modelu na {len(X_train)} rekordach...")

    # 4. Inicjalizacja i trening modelu Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1 # Wykorzystaj wszystkie rdzenie procesora
    )
    
    model.fit(X_train, y_train)

    # 5. Ewaluacja
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"✅ Model wytrenowany!")
    print(f"📈 Wyniki na zbiorze testowym:")
    print(f"   - MAE (Średni błąd): {mae:.2f} EUR/MWh")
    print(f"   - R2 Score: {r2:.4f}")

    # 6. Zapis modelu
    joblib.dump(model, MODEL_DIR / "model.joblib")
    print(f"💾 Model zapisany w: {MODEL_DIR / 'model.joblib'}")

if __name__ == "__main__":
    train_energy_model()