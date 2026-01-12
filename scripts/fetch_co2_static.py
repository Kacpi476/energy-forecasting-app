import pandas as pd
from pathlib import Path

def fetch_co2(start, end):
    """
    Pobiera dane CO2 z Twojego pliku CSV. 
    To najbezpieczniejsza metoda dla licencjatu.
    """
    csv_path = Path("data_csv/prices_eu_ets.csv")
    
    if not csv_path.exists():
        print(f"⚠ Nie znaleziono pliku {csv_path}!")
        return pd.DataFrame()

    print(f"📖 Wczytywanie danych CO2 z pliku lokalnego: {csv_path}")
    
    # Wczytanie danych (dostosuj separator jeśli trzeba, np. sep=';')
    df = pd.read_csv(csv_path)
    
    # Mapowanie kolumn i konwersja daty
    df = df[['date', 'price']].rename(columns={'price': 'co2_price_eur'})
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    
    # Ustawienie indeksu i strefy czasowej
    df = df.set_index('date').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    # Zwracamy tylko zakres, o który prosi update_file
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask]