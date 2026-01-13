import pandas as pd
from pathlib import Path

def process_my_co2_csv(file_path):
    df = pd.read_csv(file_path)
    
    df = df[['date', 'price']].rename(columns={
        'date': 'date',
        'price': 'co2_price_eur'
    })
    
    df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.tz_localize('UTC')
    
    df = df.set_index('date').sort_index()
    
    return df

my_co2 = process_my_co2_csv("data_csv/prices_eu_ets (4).csv")
my_co2.to_parquet("data/co2.parquet")