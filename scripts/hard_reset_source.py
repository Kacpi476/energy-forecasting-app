import pandas as pd
from pathlib import Path

def hard_reset_source_file():
    DATA_DIR = Path("data")
    FINAL_DATA_PATH = DATA_DIR / "final_training_data.parquet"
    
    if not FINAL_DATA_PATH.exists():
        print(f"Nie znaleziono pliku {FINAL_DATA_PATH}")
        return

    df = pd.read_parquet(FINAL_DATA_PATH)
    
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    cutoff = pd.Timestamp("2026-02-26", tz='UTC')
    
    affected_rows = len(df[df['date'] >= cutoff])
    
    df.loc[df['date'] >= cutoff, 'price_eur_mwh'] = None
    
    df.to_parquet(FINAL_DATA_PATH, index=False)
    
    print(f"--- FIZYCZNY RESET DANYCH ---")
    print(f"Plik: {FINAL_DATA_PATH}")
    print(f"Usunięto ceny dla {affected_rows} godzin (od {cutoff} w górę).")
    print(f"Ostatnia dostępna cena w pliku to teraz: {df[df['price_eur_mwh'].notna()]['date'].max()}")

if __name__ == "__main__":
    hard_reset_source_file()