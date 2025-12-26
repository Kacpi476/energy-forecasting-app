from entsoe import EntsoePandasClient
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

def fetch_demand(start, end):
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if start.tzinfo is None:
        start = start.tz_localize("UTC").tz_convert("Europe/Brussels")
    else:
        start = start.tz_convert("Europe/Brussels")

    if end.tzinfo is None:
        end = end.tz_localize("UTC").tz_convert("Europe/Brussels")
    else:
        end = end.tz_convert("Europe/Brussels")

    load = client.query_load(
        country_code="PL",
        start=start,
        end=end
    )

    if isinstance(load, pd.Series):
        df = load.to_frame(name="load_mw")

    else:
        df = load.copy()

        if len(df.columns) == 1:
            df = df.rename(columns={df.columns[0]: "load_mw"})

        if "Actual Total Load" in df.columns:
            df = df[["Actual Total Load"]].rename(columns={"Actual Total Load": "load_mw"})

        if "load_mw" not in df.columns:
            df = df.rename(columns={df.columns[0]: "load_mw"})

    df.index = df.index.tz_convert("UTC")

    df = df.reset_index().rename(columns={"index": "date"}).set_index("date")

    return df.sort_index()
