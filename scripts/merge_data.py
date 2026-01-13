import pandas as pd
import numpy as np
from pathlib import Path

START_DATE = "2024-07-01"

def merge_datasets():
    DATA_DIR = Path("data")
    
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    pse = pd.read_parquet(DATA_DIR / "pse.parquet")
    weather = pd.read_parquet(DATA_DIR / "weather.parquet")
    co2 = pd.read_parquet(DATA_DIR / "co2.parquet")

    pse_h = pse.select_dtypes(include=['number']).resample('1h').mean()
    prices_h = prices.resample('1h').mean()
    weather_h = weather.select_dtypes(include=['number']).resample('1h').mean()

    df = prices_h.join([pse_h, weather_h], how='outer').sort_index()

    df = pd.merge_asof(
        df, 
        co2.sort_index(), 
        left_index=True, 
        right_index=True, 
        direction='backward'
    )

    cols_to_fill = [
        'demand', 'pv', 'wi', 'co2_price_eur', 
        'temperature_c', 'wind_speed_ms', 'solar_wm2'
    ]
    
    existing_cols = [c for c in cols_to_fill if c in df.columns]
    df[existing_cols] = df[existing_cols].ffill().bfill()

    df = df.reset_index()
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.localize('UTC')
    else:
        df['date'] = df['date'].dt.tz_convert('UTC')
    
    df = df[df['date'] >= pd.Timestamp(START_DATE, tz='UTC')]

    
    df['price_lag_24'] = df['price_eur_mwh'].shift(24).ffill()
    
    df['res_share'] = (df['pv'] + df['wi']) / df['demand'].replace(0, 1)
    
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df = df.dropna(subset=['demand', 'co2_price_eur', 'price_lag_24'])
    
    df = df.drop_duplicates(subset=['date']).sort_values('date')
    
    return df

if __name__ == "__main__":
    df_result = merge_datasets()
    print(f"Dane zmergowane pomyślnie.")
    print(f"Zakres: {df_result['date'].min()} do {df_result['date'].max()}")
    print(f"Liczba rekordów: {len(df_result)}")