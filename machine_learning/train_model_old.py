import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_parquet("data/final_training_data.parquet")

df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

features = ['demand', 'pv', 'wi', 'temperature_c', 'wind_speed_ms', 
            'solar_wm2', 'co2_price_eur', 'hour', 'day_of_week', 'month']
target = 'price_eur_mwh'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=12, 
    random_state=42, 
    n_jobs=-1
)

print("Trenowanie modelu Random Forest...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"WYNIKI MODELU:")
print(f"Średni błąd (MAE): {mae:.2f} EUR/MWh")
print(f"Dopasowanie (R2): {r2:.2f}")
print("-" * 30)

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='skyblue')
plt.title("Wpływ zmiennych na cenę energii w Polsce (Feature Importance)")
plt.xlabel("Waga cechy w modelu")
plt.tight_layout()
plt.show()