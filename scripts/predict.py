import pandas as pd
import joblib
from pathlib import Path

def get_latest_forecast():
    DATA_PATH = Path("data/final_training_data.parquet")
    MODEL_PATH = Path("models/price_rf_model.pkl")
    FEATURES_PATH = Path("models/feature_names.pkl")

    df = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['price_lag_1'] = df['price_eur_mwh'].shift(1)
    df['res_share'] = (df['pv'] + df['wi']) / df['demand']
    
    latest_row = df.tail(1)
    
    if latest_row[features].isnull().values.any():
        latest_row = df.tail(2).head(1)

    prediction = model.predict(latest_row[features])[0]
    
    return prediction, latest_row['date'].iloc[0]