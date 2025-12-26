import requests
import pandas as pd
import ast

url = (
    "https://api.raporty.pse.pl/api/his-wlk-cal"
    "?$filter=business_date eq '2024-11-29'"
    "&$orderby=dtime_utc asc"
    "&$first=20000"
)

OUTPUT_CSV = "pse_clean_2024-11-29-2.csv"

TARGET_COLUMNS = ["dtime_utc", "demand", "pv", "wi", "jg", "jnwrb"]

def pobierz_i_wyczysc(url, output_file=OUTPUT_CSV):
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)

    df = df.applymap(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else x)

    rows = []
    for _, row in df.iterrows():
        if isinstance(row[0], dict):
            rows.append(row[0])
        else:
            rows.append(row)

    df2 = pd.DataFrame(rows)

    df2 = df2[TARGET_COLUMNS].copy()

    df2["dtime_utc"] = pd.to_datetime(df2["dtime_utc"])

    df2 = df2.sort_values("dtime_utc")

    df2.to_csv(output_file, index=False)
    print("✔️ Zapisano oczyszczony plik:", output_file)

    return df2


if __name__ == "__main__":
    df = pobierz_i_wyczysc(url)
    print(df.head())
