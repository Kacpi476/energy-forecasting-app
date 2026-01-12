import pandas as pd
from pathlib import Path

START_DATE = "2024-07-01"

def merge_datasets():
    DATA_DIR = Path("data")
    
    # 1. Wczytanie danych
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    pse = pd.read_parquet(DATA_DIR / "pse.parquet")
    weather = pd.read_parquet(DATA_DIR / "weather.parquet")
    co2 = pd.read_parquet(DATA_DIR / "co2.parquet")

    # 2. Ujednolicenie indeksów do 1h przed mergem
    # PSE jest co 15 min -> średnia godzinowa
    pse_h = pse.select_dtypes(include=['number']).resample('1h').mean()
    
    # Ceny i Pogoda zazwyczaj są co 1h, ale robimy resample dla pewności
    prices_h = prices.resample('1h').mean()
    weather_h = weather.select_dtypes(include=['number']).resample('1h').mean()

    # 3. Łączenie główne (Prices + PSE + Weather) - wszystko już jest co 1h
    df = prices_h.join([pse_h, weather_h], how='left').sort_index()

    # 4. Inteligentne łączenie z CO2 (asof merge)
    # CO2 jest zazwyczaj raz na dobę, więc backward fill pasuje idealnie
    df = pd.merge_asof(
        df, 
        co2.sort_index(), 
        left_index=True, 
        right_index=True, 
        direction='backward'
    )

    # 5. Uzupełnienie braków
    # ffill() wypełnia ewentualne braki w OZE/pogodzie/CO2 wartością z poprzedniej godziny
    potential_features = ['demand', 'pv', 'wi', 'co2_price_eur', 'temp', 'wind_speed', 'temperature', 'windspeed']
    existing_cols = [c for c in potential_features if c in df.columns]
    df[existing_cols] = df[existing_cols].ffill().bfill()

    # 6. Czyszczenie i Feature Engineering
    df = df.reset_index()
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    
    # Filtrowanie zakresu
    df = df[df['date'] >= pd.Timestamp(START_DATE, tz='UTC')]

    # Tworzenie cech pod ML
    df['price_lag_1'] = df['price_eur_mwh'].shift(1)
    df['res_share'] = (df['pv'] + df['wi']) / df['demand']
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek

    # Usuwamy tylko wiersze krytyczne (np. pierwszy po shift)
    df = df.dropna(subset=['demand', 'co2_price_eur', 'price_lag_1'])
    
    # Finalna weryfikacja - upewniamy się, że nie ma duplikatów godzin
    df = df.drop_duplicates(subset=['date']).sort_values('date')
    
    return df