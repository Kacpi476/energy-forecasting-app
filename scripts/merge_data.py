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

    # resampling do 1h
    pse_h = pse.select_dtypes(include=['number']).resample('1h').mean()
    prices_h = prices.resample('1h').mean()
    weather_h = weather.select_dtypes(include=['number']).resample('1h').mean()

    df = prices_h.join([pse_h, weather_h], how='outer').sort_index()
    df = pd.merge_asof(df, co2.sort_index(), left_index=True, right_index=True, direction='backward')

    df = df.reset_index()
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.localize('UTC')
    else:
        df['date'] = df['date'].dt.tz_convert('UTC')


    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek

    history = df[df['price_eur_mwh'].notna()].copy()
    demand_profile = history.groupby(['day_of_week', 'hour'])['demand'].mean().reset_index()
    demand_profile.rename(columns={'demand': 'typical_demand'}, inplace=True)
    
    df = df.merge(demand_profile, on=['day_of_week', 'hour'], how='left')
    df['demand'] = df['demand'].fillna(df['typical_demand'])
    df.drop(columns=['typical_demand'], inplace=True)

    cols_to_fill = ['co2_price_eur', 'temperature_c', 'wind_speed_ms', 'solar_wm2', 'pv', 'wi']
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill()

    df.loc[(df['hour'] < 6) | (df['hour'] > 19), 'pv'] = 0

    # CECHY KOŃCOWE
    df['price_lag_24'] = df['price_eur_mwh'].shift(24)
    df['res_share'] = (df['pv'] + df['wi']) / df['demand'].replace(0, 1)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df = df[df['date'] >= pd.Timestamp(START_DATE, tz='UTC')]
    
    essential_features = ['demand', 'co2_price_eur', 'temperature_c']
    df = df.dropna(subset=essential_features)
    
    return df

if __name__ == "__main__":
    df_result = merge_datasets()
    df_result.to_parquet("data/final_training_data.parquet", index=False)
    
    last_price = df_result[df_result['price_eur_mwh'].notna()]['date'].max()
    print(f"Dane zmergowane.")
    print(f"Ostatnia cena rzeczywista: {last_price}")
    print(f"Liczba godzin do prognozowania: {df_result[df_result['price_eur_mwh'].isna()].shape[0]}")