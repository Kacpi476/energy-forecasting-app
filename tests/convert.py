import pandas as pd

df = pd.read_parquet("data/prices.parquet")
df.to_csv("data_csv/prices.csv")

#df2 = pd.read_parquet("data/demand.parquet")
#df2.to_csv("data_csv/demand.csv")

df3 = pd.read_parquet("data/co2.parquet")
df3.to_csv("data_csv/co2.csv")

df4 = pd.read_parquet("data/weather.parquet")
df4.to_csv("data_csv/weather.csv")

#df5 = pd.read_parquet("data/recent_demand.parquet")
#df5.to_csv("data_csv/recent_demand.csv")

#df6 = pd.read_csv("data_test/pse_full_history_clean.csv")
#df6.to_parquet("data/pse_full_history_clean.parquet")

df7 = pd.read_parquet("data/pse.parquet")
df7.to_csv("data_csv/pse.csv")


df8 = pd.read_parquet("data/final_training_data.parquet")
df8.to_csv("data_csv/final_training_data.csv")

df9 = pd.read_parquet("data/forecast_history.parquet")
df9.to_csv("data_csv/forecast_history.csv")
