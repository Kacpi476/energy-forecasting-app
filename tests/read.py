import pandas as pd

# Wczytanie JSONa
df_json = pd.read_json("data/latest_forecast.json")

print("📊 PODGLĄD PROGNOZY Z JSON:")
print(df_json.head())
print("\n🔍 Sprawdzenie typów danych:")
print(df_json.dtypes)