import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 1. Wczytaj model i nazwy cech
model = joblib.load("models/price_rf_model.pkl")
features = joblib.load("models/feature_names.pkl")

# 2. Wyciągnij ważność cech
importances = model.feature_importances_

# 3. Zrób z tego ładną tabelkę
feature_importance_df = pd.DataFrame({
    'Cecha': features,
    'Ważność': importances
}).sort_values(by='Ważność', ascending=False)

print(feature_importance_df)

# 4. (Opcjonalnie) Wykres
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Cecha'], feature_importance_df['Ważność'])
plt.gca().invert_yaxis()
plt.title("Ważność cech w modelu Random Forest")
plt.show()