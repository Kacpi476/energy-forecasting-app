import pandas as pd
from pathlib import Path

def process_my_co2_csv(file_path):
    # Wczytujemy plik (uważaj na separator, jeśli to polski Excel to może być średnik ';')
    df = pd.read_csv(file_path)
    
    # Zmiana nazw kolumn na docelowe
    # Zakładamy kolejność: date, indeks, price, volume
    df = df[['date', 'price']].rename(columns={
        'date': 'date',
        'price': 'co2_price_eur'
    })
    
    # Konwersja daty - Twoje dane mają format DD.MM.YYYY
    df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.tz_localize('UTC')
    
    # Ustawienie indeksu
    df = df.set_index('date').sort_index()
    
    return df

# Użycie:
my_co2 = process_my_co2_csv("data_csv/prices_eu_ets (4).csv")
my_co2.to_parquet("data/co2.parquet")