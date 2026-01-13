import pandas as pd

# Wczytanie JSONa
df_json = pd.read_json("data/latest_forecast.json")

print("PODGLĄD PROGNOZY Z JSON:")
print(df_json.head())
print("\nSprawdzenie typów danych:")
print(df_json.dtypes)