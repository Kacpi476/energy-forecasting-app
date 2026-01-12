import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_parquet("data/final_training_data.parquet")

analysis_cols = [
    'price_eur_mwh',
    'co2_price_eur',
    'demand', 
    'wi',
    'pv',
    'temperature_c'
]

corr_matrix = df[analysis_cols].corr()

print("MACIERZ KORELACJI (Pearson):")
print(corr_matrix['price_eur_mwh'].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=0.5)
plt.title("Korelacja czynników z ceną energii w Polsce")
plt.tight_layout()
plt.savefig("korelacja_heatmapa.png")
plt.show()

plt.figure(figsize=(8, 6))
sns.regplot(data=df, x='co2_price_eur', y='price_eur_mwh', 
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title("Zależność ceny energii od ceny CO2")
plt.xlabel("Cena CO2 [EUR/t]")
plt.ylabel("Cena energii [EUR/MWh]")
plt.tight_layout()
plt.show()