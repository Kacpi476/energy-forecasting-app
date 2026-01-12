import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Wczytanie danych
df = pd.read_parquet("data/final_training_data.parquet")

# 2. Inżynieria cech czasowych
# Model musi wiedzieć, która jest godzina i dzień
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# 3. Definiujemy cechy (X) i cel (y)
# Wybieramy najważniejsze kolumny do modelu
features = ['demand', 'pv', 'wi', 'temperature_c', 'wind_speed_ms', 
            'solar_wm2', 'co2_price_eur', 'hour', 'day_of_week', 'month']
target = 'price_eur_mwh'

X = df[features]
y = df[target]

# 4. Podział na zbiór treningowy i testowy
# Dla szeregów czasowych NIE używamy shuffle=True! 
# Uczymy się na przeszłości, testujemy na przyszłości.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. Budowa modelu Random Forest
# n_jobs=-1 sprawi, że Mac użyje wszystkich rdzeni procesora
model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=12, 
    random_state=42, 
    n_jobs=-1
)

print("🚀 Trenowanie modelu Random Forest...")
model.fit(X_train, y_train)

# 6. Predykcja i wyniki
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"📊 WYNIKI MODELU:")
print(f"Średni błąd (MAE): {mae:.2f} EUR/MWh")
print(f"Dopasowanie (R2): {r2:.2f}")
print("-" * 30)

# 7. WYKRES: Ważność cech (Klucz do Twojego licencjatu!)
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='skyblue')
plt.title("Wpływ zmiennych na cenę energii w Polsce (Feature Importance)")
plt.xlabel("Waga cechy w modelu")
plt.tight_layout()
plt.show()