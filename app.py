import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Prognozowanie Cen Energii | Licencjat", 
    layout="wide",
)

@st.cache_data
def load_data():
    """Wczytuje dane i filtruje od 8 stycznia 2026."""
    # Definiujemy datę startową
    START_DISPLAY = pd.Timestamp("2026-01-08", tz='UTC')
    
    # 1. Dane rzeczywiste
    df_hist = pd.read_parquet("data/final_training_data.parquet")
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    if df_hist['date'].dt.tz is None:
        df_hist['date'] = df_hist['date'].dt.localize('UTC')
    
    # 2. Historia prognoz
    forecast_path = Path("data/forecast_history.parquet")
    if forecast_path.exists():
        df_forecast_all = pd.read_parquet(forecast_path)
        df_forecast_all['date'] = pd.to_datetime(df_forecast_all['date'])
        if df_forecast_all['date'].dt.tz is None:
            df_forecast_all['date'] = df_forecast_all['date'].dt.localize('UTC')
    else:
        df_forecast_all = pd.DataFrame()

    # Filtrowanie od 8 stycznia 2026
    df_hist = df_hist[df_hist['date'] >= START_DISPLAY]
    if not df_forecast_all.empty:
        df_forecast_all = df_forecast_all[df_forecast_all['date'] >= START_DISPLAY]
    
    return df_hist, df_forecast_all

# --- GŁÓWNA LOGIKA APLIKACJI ---

try:
    df_hist, df_forecast_all = load_data()
    
    st.title("⚡ Analiza Porównawcza: Rzeczywistość vs Model")
    st.subheader("Okres analizy: od 8 stycznia 2026")

    # Wykres z porównaniem
    fig = go.Figure()

    # Linia rzeczywista
    fig.add_trace(go.Scatter(
        x=df_hist['date'], y=df_hist['price_eur_mwh'],
        name="Cena rzeczywista (TGE)",
        line=dict(color='#1f77b4', width=4)
    ))

    # Linia prognozowana
    if not df_forecast_all.empty:
        fig.add_trace(go.Scatter(
            x=df_forecast_all['date'], y=df_forecast_all['predicted_price'],
            name="Historyczna prognoza modelu",
            line=dict(color='#d62728', width=2, dash='dot')
        ))

    fig.update_layout(
        title="Zestawienie cen energii [EUR/MWh]",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        xaxis=dict(range=[pd.Timestamp("2026-01-08", tz='UTC'), df_forecast_all['date'].max() if not df_forecast_all.empty else None])
    )
    
    # Linia "TERAZ" dla orientacji (dziś jest 13 stycznia)
    fig.add_vline(x=pd.Timestamp.now(tz='UTC'), line_width=2, line_dash="solid", line_color="green")
    
    st.plotly_chart(fig, use_container_width=True)

    # Sekcja statystyk błędu dla tego konkretnego okresu
    st.divider()
    st.header("Statystyki błędu od 8 stycznia")
    
    merged = pd.merge(df_hist[['date', 'price_eur_mwh']], df_forecast_all[['date', 'predicted_price']], on='date').dropna()
    
    if not merged.empty:
        merged['error'] = merged['price_eur_mwh'] - merged['predicted_price']
        mae = merged['error'].abs().mean()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (Średni błąd)", f"{mae:.2f} EUR")
        c2.metric("Max błąd", f"{merged['error'].abs().max():.2f} EUR")
        c3.metric("Liczba próbek", f"{len(merged)}")
        
        st.write("Wykres rozrzutu błędów:")
        fig_err = px.histogram(merged, x="error", nbins=20, title="Rozkład błędu prognozy [EUR]", color_discrete_sequence=['indianred'])
        st.plotly_chart(fig_err, use_container_width=True)
    else:
        st.warning("Brak pokrywających się danych (rzeczywiste vs prognozy) dla tego okresu.")

except Exception as e:
    st.error(f"Wystąpił błąd: {e}")