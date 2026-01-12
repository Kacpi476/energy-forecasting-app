from entsoe import EntsoePandasClient
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
start = pd.Timestamp("2024-11-01", tz="Europe/Brussels")
end   = pd.Timestamp("2024-11-03", tz="Europe/Brussels")

df = client.query_load(country_code="PL", start=start, end=end)
print(df.head(), df.index.min(), df.index.max())
print(len(df))