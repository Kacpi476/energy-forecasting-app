import pandas as pd
df = pd.read_parquet("data/forecast_history.parquet")
df['date'] = pd.to_datetime(df['date'], utc=True)
print("Min date:", df['date'].min())
print("Predicted notna:", df['predicted_price'].notna().sum())
print("Price notna:", df['price_eur_mwh'].notna().sum())
print("Overlap (oba notna):", (df['predicted_price'].notna() & df['price_eur_mwh'].notna()).sum())