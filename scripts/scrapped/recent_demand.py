from entsoe import EntsoePandasClient
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

def fetch_recent_demand_week():
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)

    df = client.query_load(
        country_code="PL",
        start=pd.Timestamp(week_ago, tz="Europe/Brussels"),
        end=pd.Timestamp(today + timedelta(days=1), tz="Europe/Brussels")
    )

    if isinstance(df, pd.Series):
        df = df.to_frame(name="load_mw")
    else:
        df.columns = ["load_mw"]  

    df.index = df.index.tz_convert("UTC")

    df_daily = df.resample("D").sum()
    df_daily = df_daily.reset_index().rename(columns={"index": "date"}).set_index("date")

    return df_daily

if __name__ == "__main__":
    df = fetch_recent_demand_week()
    print(df)
    df.to_parquet("data/recent_demand.parquet")
