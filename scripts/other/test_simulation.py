import pandas as pd
import joblib
from pathlib import Path

def run_simulation_test():
    DATA_DIR = Path("data")
    MODEL_PATH = Path("models/price_rf_model.pkl")
    FEATURES_PATH = Path("models/feature_names.pkl")
    
    # 1. Wczytujemy dane i model
    df = pd.read_parquet(DATA_DIR / "final_training_data.parquet")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date').reset_index(drop=True)

    # --- SYMULACJA: Kasujemy ceny od 26 lutego 00:00 w górę ---
    cutoff_date = pd.Timestamp("2026-02-26", tz='UTC')
    
    # Tworzymy kopię do testu, gdzie ceny 'znikają' po 25 lutym
    test_df = df.copy()
    test_df.loc[test_df['date'] >= cutoff_date, 'price_eur_mwh'] = None

    print(f"🕵️ Symulacja: Dane cenowe urywają się o: {test_df[test_df['price_eur_mwh'].notna()]['date'].max()}")

    # --- LOGIKA GENEROWANIA PROGNOZY ---
    # 2. Generujemy lagi (price_lag_24)
    # Nawet jeśli nie ma ceny 26.02, to lag_24 weźmie cenę z 25.02 (która wciąż tam jest!)
    test_df['price_lag_24'] = test_df['price_eur_mwh'].shift(24)

    # 3. Wybieramy dane do prognozy (cały 26 luty)
    target_mask = test_df['date'].dt.date == cutoff_date.date()
    forecast_input = test_df[target_mask].copy()

    if forecast_input.empty:
        print("❌ Błąd: Brak danych (pogody/popytu) na 26 lutego w pliku!")
        return

    # 4. Predykcja
    X = forecast_input[features].ffill().bfill()
    predictions = model.predict(X)
    
    # 5. Wpisujemy prognozę z powrotem
    test_df.loc[target_mask, 'predicted_price'] = predictions

    # --- WERYFIKACJA ---
    print(f"\n✅ Test zakończony sukcesem!")
    print(f"📈 Wygenerowano prognozę na {len(predictions)} godzin dnia 26 lutego.")
    print(f"💰 Średnia prognozowana cena na ten dzień: {predictions.mean():.2f} EUR")

    # Zapisujemy jako specjalny plik, żebyś mógł go podejrzeć (np. w Streamlit)
    test_df[['date', 'price_eur_mwh', 'predicted_price']].to_parquet(DATA_DIR / "forecast_history.parquet")
    print(f"🚀 Wynik testu zapisany do 'forecast_history.parquet'. Odśwież dashboard!")

if __name__ == "__main__":
    run_simulation_test()