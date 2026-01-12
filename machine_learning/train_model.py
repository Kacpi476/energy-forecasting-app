import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

Path("models").mkdir(exist_ok=True)

def train_price_model():
    df = pd.read_parquet("data/final_training_data.parquet")
    
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    
    df['price_lag_1'] = df['price_eur_mwh'].shift(1)
    
    df['res_share'] = (df['pv'] + df['wi']) / df['demand']
    
    df = df.dropna()

    features = ['demand', 'pv', 'wi', 'temperature_c', 'co2_price_eur', 
                'hour', 'day_of_week', 'price_lag_1', 'res_share']
    target = 'price_eur_mwh'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    model = RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"📊 Wyniki po poprawkach:")
    print(f"Błąd MAE: {mean_absolute_error(y_test, preds):.2f} EUR")
    print(f"Dopasowanie R2: {r2_score(y_test, preds):.2f}")

    joblib.dump(model, 'models/price_rf_model.pkl')
    joblib.dump(features, 'models/feature_names.pkl')
    print("✅ Model zapisany w 'models/price_rf_model.pkl'")

if __name__ == "__main__":
    train_price_model()