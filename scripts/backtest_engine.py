import pandas as pd
import joblib
from pathlib import Path

def run_backtest():
    DATA_DIR = Path("data")
    MODEL_PATH = Path("models/price_rf_model.pkl")
    FEATURES_PATH = Path("models/feature_names.pkl")
    
    df = pd.read_parquet(DATA_DIR / "final_training_data.parquet")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date').reset_index(drop=True)


    df['day'] = df['date'].dt.date

    price_counts = df[df['price_eur_mwh'].notna()].groupby('day').size()
    full_days = price_counts[price_counts == 24].index.tolist()

    if not full_days:
        print("Brak pełnych dób (24h) z cenami w pliku!")
        return

    last_full_day = max(full_days)
    
    forecast_target_day = last_full_day + pd.Timedelta(days=1)

    print(f"Ostatni pełny dzień danych: {last_full_day}")
    print(f"Generuję prognozę na całą dobę: {forecast_target_day}")

    forecast_mask = df['day'] == forecast_target_day
    if df[forecast_mask].shape[0] < 24:
        print(f"Brak pełnych danych pogodowych na {forecast_target_day}")
        return

    plot_df = df[df['day'] <= forecast_target_day].copy()

    plot_df['price_lag_24'] = plot_df['price_eur_mwh'].shift(24)

    X = plot_df[features].ffill().bfill()
    plot_df['predicted_price'] = model.predict(X)

    output_df = plot_df[['date', 'price_eur_mwh', 'predicted_price']]
    output_df.to_parquet(DATA_DIR / "forecast_history.parquet")

    print(f"Wygenerowano prognozę.")
    print(f"Zakres wykresu: {output_df['date'].min().date()} do {output_df['date'].max().date()}")
    print(f"Jutro ({forecast_target_day}) ma {len(output_df[output_df['date'].dt.date == forecast_target_day])}h prognozy.")

if __name__ == "__main__":
    run_backtest()