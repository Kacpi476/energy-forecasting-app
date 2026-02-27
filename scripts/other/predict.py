import pandas as pd
import joblib
from pathlib import Path

def run_predict():
    DATA_DIR = Path("data")
    model = joblib.load("models/price_rf_model.pkl")
    features = joblib.load("models/feature_names.pkl")
    
    df = pd.read_parquet(DATA_DIR / "final_training_data.parquet")
    df['date'] = pd.to_datetime(df['date'], utc=True)

    # Szukamy tylko dziur (NaN) w cenach
    future_df = df[df['price_eur_mwh'].isna()].copy()
    
    if not future_df.empty:
        X = future_df[features].ffill().bfill()
        future_df['predicted_price_eur'] = model.predict(X)
        future_df[['date', 'predicted_price_eur']].to_csv(DATA_DIR / "forecast_output.csv", index=False)
        print(f"🔮 Wygenerowano prognozę na {len(future_df)}h przyszłości.")
    else:
        print("Brak dziur do wypełnienia.")

if __name__ == "__main__":
    run_predict()