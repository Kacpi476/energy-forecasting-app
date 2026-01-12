import pandas as pd
import joblib
from pathlib import Path

def generate_forecasts():
    DATA_DIR = Path("data")
    MODEL_PATH = Path("machine_learning/model.joblib")
    FINAL_DATA_PATH = DATA_DIR / "final_training_data.parquet"
    FORECAST_HISTORY_PATH = DATA_DIR / "forecast_history.parquet"

    if not MODEL_PATH.exists():
        print("❌ Błąd: Brak wytrenowanego modelu w machine_learning/model.joblib")
        return

    # 1. Wczytanie danych i modelu
    df = pd.read_parquet(FINAL_DATA_PATH)
    model = joblib.load(MODEL_PATH)
    
    # Definicja cech (musi być identyczna jak przy treningu!)
    features = ['demand', 'pv', 'wi', 'co2_price_eur', 'hour', 'day_of_week', 'res_share', 'price_lag_1']
    
    # 2. Wybieramy wiersze do prognozowania
    # Są to wiersze, gdzie price_eur_mwh jest NaN (dane przyszłe) 
    # LUB po prostu bierzemy ostatnie 48h, żeby mieć pewność, że pokrywamy "dziś i jutro"
    to_predict = df[features].tail(48) # Prognozujemy ostatnie 2 dni
    
    # 3. Generowanie prognoz
    predictions = model.predict(to_predict)
    
    # 4. Przygotowanie ramki wynikowej
    new_forecasts = pd.DataFrame({
        'date': df.loc[to_predict.index, 'date'],
        'predicted_price': predictions,
        'forecast_made_at': pd.Timestamp.now(tz='UTC')
    })

    # 5. Zapis/Aktualizacja historii
    if FORECAST_HISTORY_PATH.exists():
        old_forecasts = pd.read_parquet(FORECAST_HISTORY_PATH)
        # Łączymy i usuwamy duplikaty (zachowujemy najnowszą prognozę dla danej godziny)
        combined = pd.concat([old_forecasts, new_forecasts])
        combined = combined.drop_duplicates(subset=['date'], keep='last').sort_values('date')
        combined.to_parquet(FORECAST_HISTORY_PATH)
    else:
        new_forecasts.to_parquet(FORECAST_HISTORY_PATH)

    print(f"🔮 Wygenerowano prognozy dla zakresu: {new_forecasts['date'].min()} do {new_forecasts['date'].max()}")
    return new_forecasts

if __name__ == "__main__":
    generate_forecasts()