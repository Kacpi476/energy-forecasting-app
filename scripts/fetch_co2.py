import pandas as pd
import yfinance as yf
from datetime import timedelta

def fetch_co2(start, end):
    try:
        ticker = "KEUA" 
        fetch_end = end + timedelta(days=2)

        data = yf.download(ticker, start=start, end=fetch_end, interval="1d", progress=False)
        
        if data.empty:
            data = yf.download("KRBN", start=start, end=fetch_end, interval="1d", progress=False)

        if data.empty:
            return pd.DataFrame()

        df = data[['Close']].copy()
        df.columns = ['co2_price_eur']
        df.index.name = 'date'
        
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index).tz_localize("UTC")
        else:
            df.index = pd.to_datetime(df.index).tz_convert("UTC")

        df = df.dropna().sort_index()

        print(f"Pobrano dane CO2. Ostatnia cena: {df['co2_price_eur'].iloc[-1]:.2f} USD/jednostkę")
        
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask]

    except Exception as e:
        print(f"Błąd: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    from datetime import datetime, timezone
    s = datetime(2025, 1, 1, tzinfo=timezone.utc)
    e = datetime.now(timezone.utc)
    print(fetch_co2(s, e))