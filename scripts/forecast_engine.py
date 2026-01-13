import pandas as pd
import joblib
from pathlib import Path

def generate_forecasts():
    DATA_DIR = Path("data")
    MODEL_PATH = Path("machine_learning/model.joblib")
    FINAL_DATA_PATH = DATA_DIR / "final_training_data.parquet"
    FORECAST_HISTORY_PATH = DATA_DIR / "forecast_history.parquet"

    if not MODEL_PATH.exists():
        print("Błąd: Brak wytrenowanego modelu.")
        return None

    df = pd.read_parquet(FINAL_DATA_PATH)
    model = joblib.load(MODEL_PATH)
    
    features = [
        'demand', 'pv', 'wi', 'co2_price_eur', 
        'temperature_c', 'wind_speed_ms', 'solar_wm2',
        'hour_sin', 'hour_cos', 'day_of_week', 
        'res_share', 'price_lag_24'
    ]
    
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert('UTC') if df['date'].dt.tz else pd.to_datetime(df['date']).dt.tz_localize('UTC')

    to_predict = df.tail(168).copy() # Prognozujemy ostatnie 7 dni + przyszłość dla wykresu

    print(f"Generowanie prognoz dla {len(to_predict)} punktów...")
    
    X = to_predict[features]
    predictions = model.predict(X)
    
    new_forecasts = pd.DataFrame({
        'date': to_predict['date'].values,
        'predicted_price': predictions,
        'forecast_made_at': pd.Timestamp.now(tz='UTC')
    })
    
    new_forecasts['date'] = pd.to_datetime(new_forecasts['date'])
    
    if new_forecasts['date'].dt.tz is None:
        new_forecasts['date'] = new_forecasts['date'].dt.tz_localize('UTC')
    else:
        new_forecasts['date'] = new_forecasts['date'].dt.tz_convert('UTC')

    if FORECAST_HISTORY_PATH.exists():
        old_history = pd.read_parquet(FORECAST_HISTORY_PATH)
        old_history['date'] = pd.to_datetime(old_history['date']).dt.tz_convert('UTC') if old_history['date'].dt.tz else pd.to_datetime(old_history['date']).dt.tz_localize('UTC')
        combined = pd.concat([old_history, new_forecasts])
    else:
        combined = new_forecasts

    combined = combined.drop_duplicates(subset=['date'], keep='last').sort_values('date')
    
    combined.to_parquet(FORECAST_HISTORY_PATH)
    
    combined.tail(48).to_json(DATA_DIR / "latest_forecast.json", orient='records', date_format='iso')

    print(f"Sukces! Najdalsza prognoza: {combined['date'].max()}")
    return combined

if __name__ == "__main__":
    generate_forecasts()