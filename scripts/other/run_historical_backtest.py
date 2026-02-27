import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm

def run_historical_backtest():
    DATA_DIR = Path("data")
    MODEL_PATH = Path("models/price_rf_model.pkl")
    FEATURES_PATH = Path("models/feature_names.pkl")
    
    # 1. Wczytujemy dane i model
    df = pd.read_parquet(DATA_DIR / "final_training_data.parquet")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Przygotowujemy kolumnę na wyniki backtestu
    df['predicted_price'] = None 
    
    # 2. Definiujemy zakres: od 1 stycznia 2026 do dzisiaj
    start_date = pd.Timestamp("2026-01-01", tz='UTC').date()
    end_date = df['date'].max().date()
    
    all_days = pd.date_range(start_date, end_date, freq='D')
    
    print(f"🚀 Rozpoczynam historyczny backtest od {start_date} do {end_date}...")

    # 3. Pętla dzień po dniu
    for current_day in tqdm(all_days):
        # Sprawdzamy, czy poprzedni dzień (T-1) ma komplet 24h danych (cen)
        yesterday = current_day - pd.Timedelta(days=1)
        yesterday_data = df[df['date'].dt.date == yesterday.date()]
        
        # WARUNEK: Robimy prognozę na 'current_day' tylko jeśli 'yesterday' ma komplet cen
        if yesterday_data['price_eur_mwh'].notna().sum() == 24:
            
            # Pobieramy dane na dzień, który chcemy prognozować
            target_day_mask = df['date'].dt.date == current_day.date()
            target_data = df[target_day_mask].copy()
            
            if not target_data.empty:
                # Generujemy lagi dla tego konkretnego dnia na podstawie 'yesterday'
                # (W backteście możemy użyć shift(24) na całym DF, co jest szybsze)
                X = target_data[features].ffill().bfill()
                
                # Zapisujemy prognozę w głównym DataFrame
                df.loc[target_day_mask, 'predicted_price'] = model.predict(X)

    # 4. ZAPIS WYNIKÓW
    # Zapisujemy tylko okres od 1 stycznia, żeby nie puchły pliki
    output_df = df[df['date'].dt.date >= start_date].copy()
    output_df.to_parquet(DATA_DIR / "forecast_history.parquet")
    
    print(f"\n✅ Backtest zakończony!")
    print(f"📊 Wygenerowano prognozy dla {output_df['predicted_price'].notna().sum() // 24} dni.")

if __name__ == "__main__":
    run_historical_backtest()