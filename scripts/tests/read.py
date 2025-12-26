import pandas as pd
df = pd.read_parquet("data/pse_full_history_clean.parquet")
print(df.head())
print(df.index.tz)   
print(df.resample("D").mean().head()) 
