import pandas as pd
import joblib
from pathlib import Path

def generate_forecasts():
    DATA_DIR = Path("data")
    MODEL_PATH = Path("machine_learning/model.joblib")
    FINAL_DATA_PATH = DATA_DIR / "final_training_data.parquet"
    FORECAST_HISTORY_PATH = DATA_DIR / "forecast_history.parquet"

    if not MODEL_PATH.exists():
        print("❌ Błąd: Brak modelu.")
        return None

    df = pd.read_parquet(FINAL_DATA_PATH)
    model = joblib.load(MODEL_PATH)
    
    features = [
        'demand', 'pv', 'wi', 'co2_price_eur', 
        'temperature_c', 'wind_speed_ms', 'solar_wm2',
        'hour_sin', 'hour_cos', 'day_of_week', 
        'res_share', 'price_lag_24'
    ]
    
    # 1. Standaryzacja dat
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    else:
        df['date'] = df['date'].dt.tz_convert('UTC')

    # 2. LOGIKA KROCZĄCA: Znajdź ostatni dzień z ceną rzeczywistą
    # Szukamy ostatniego wiersza, gdzie price_eur_mwh NIE jest NaN
    last_real_price_date = df[df['price_eur_mwh'].notna()]['date'].max()
    
    if pd.isna(last_real_price_date):
        print("❌ Błąd: Nie znaleziono cen rzeczywistych w danych.")
        return None

    print(f"Ostatnia cena rzeczywista z dnia: {last_real_price_date}")

    # 3. Wybieramy okno do prognozy: wszystko powyżej last_real_price_date
    # To zapewni, że prognozujemy tylko "przyszłość" względem znanych cen
    to_predict = df[df['date'] > last_real_price_date].copy()

    if to_predict.empty:
        print("✅ Brak nowych dni do prognozowania.")
        return None

    print(f"🔮 Generowanie prognoz kroczących dla {len(to_predict)} punktów...")
    
    # 4. Predykcja
    X = to_predict[features]
    predictions = model.predict(X)
    
    # 5. Budowa ramki wynikowej
    new_forecasts = pd.DataFrame({
        'date': to_predict['date'].values,
        'predicted_price': predictions,
        'forecast_made_at': pd.Timestamp.now(tz='UTC')
    })
    
    new_forecasts['date'] = pd.to_datetime(new_forecasts['date'])
    if new_forecasts['date'].dt.tz is None:
        new_forecasts['date'] = new_forecasts['date'].dt.tz_localize('UTC')

    # 6. Łączenie z historią
    if FORECAST_HISTORY_PATH.exists():
        old_history = pd.read_parquet(FORECAST_HISTORY_PATH)
        old_history['date'] = pd.to_datetime(old_history['date'])
        if old_history['date'].dt.tz is None:
            old_history['date'] = old_history['date'].dt.tz_localize('UTC')
        
        combined = pd.concat([old_history, new_forecasts])
    else:
        combined = new_forecasts

    combined = combined.drop_duplicates(subset=['date'], keep='last').sort_values('date')
    
    # Zapis
    combined.to_parquet(FORECAST_HISTORY_PATH)
    # Wysyłamy do JSON tylko 48h (dzisiaj i jutro), żeby dashboard był czytelny
    combined.tail(48).to_json(DATA_DIR / "latest_forecast.json", orient='records', date_format='iso')

    print(f"✨ Sukces! Prognoza wygenerowana do: {combined['date'].max()}")
    return combined

if __name__ == "__main__":
    generate_forecasts()