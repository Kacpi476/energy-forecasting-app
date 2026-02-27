import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm

def run_clean_backtest():
    DATA_DIR = Path("data")
    MODEL_PATH = Path("models/price_rf_model.pkl")
    FEATURES_PATH = Path("models/feature_names.pkl")
    
    # 1. Wczytujemy dane
    df = pd.read_parquet(DATA_DIR / "final_training_data.parquet")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date').reset_index(drop=True)

    # 2. PRZYGOTOWANIE POLA BITWY
    # Tworzymy kopię, w której przechowamy prawdziwe ceny do późniejszego porównania
    df_results = df[['date', 'price_eur_mwh']].copy()
    df_results['predicted_price'] = None

    # Granica testu
    test_start = pd.Timestamp("2026-01-01", tz='UTC')
    test_end = pd.Timestamp("2026-02-26", tz='UTC') # Odcinamy wszystko po północy 26.02
    
    all_days = pd.date_range(start=test_start, end=test_end, freq='D')

    print(f"🧹 Czyszczenie danych... Symulacja startuje: {test_start.date()}")

    # 3. PĘTLA DZIEŃ PO DNIU (Walk-Forward)
    for current_day in tqdm(all_days):
        day_date = current_day.date()
        
        # Maska dla dnia, który prognozujemy
        mask = df['date'].dt.date == day_date
        
        if df[mask].empty:
            continue

        # KLUCZ: Model bierze cechy (pogoda, popyt) z oryginalnego pliku,
        # ale 'price_lag_24' musi pochodzić z PRAWDZIWEJ CENY z wczoraj.
        # Pobieramy dane wejściowe dla modelu
        X_input = df.loc[mask, features].copy()
        
        # Upewniamy się, że price_lag_24 jest wypełniony (pochodzi z df, gdzie ceny są)
        X_input = X_input.ffill().bfill()
        
        # Predykcja na cały dzień (24h)
        day_preds = model.predict(X_input)
        
        # Zapisujemy wynik do naszego pliku wynikowego
        df_results.loc[mask, 'predicted_price'] = day_preds

    # 4. OSTATECZNE CIĘCIE (Zgodnie z Twoją prośbą)
    # Usuwamy ceny rzeczywiste po 26 lutym 00:00, żeby nie "oszukiwały" na wykresie
    df_results.loc[df_results['date'] >= test_end, 'price_eur_mwh'] = None

    # 5. ZAPIS
    df_results.to_parquet(DATA_DIR / "forecast_history.parquet")
    
    print(f"\n✨ Gotowe! Dane od {test_start.date()} zostały przeliczone.")
    print(f"🚫 Ceny rzeczywiste po {test_end} zostały usunięte z pliku wynikowego.")

if __name__ == "__main__":
    run_clean_backtest()