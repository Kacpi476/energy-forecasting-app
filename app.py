import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Monitoring Cen Energii")

def load_data():
    df = pd.read_parquet("data/final_training_data.parquet")
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    try:
        df_back = pd.read_parquet("data/forecast_history.parquet")
        df_back['date'] = pd.to_datetime(df_back['date'], utc=True)
    except:
        df_back = pd.DataFrame()
        
    return df, df_back

try:
    df, df_back = load_data()
    
    start_view = pd.Timestamp("2026-01-01", tz='UTC')
    now_utc = pd.Timestamp.now(tz='UTC')
    
    st.title("⚡ Analiza Modelu: Cena Realna vs Przewidywania")
    st.subheader("Okres: od 1 stycznia 2026 (w tym prognoza bieżąca)")

    fig = go.Figure()

    hist = df[df['date'] >= start_view].dropna(subset=['price_eur_mwh'])
    fig.add_trace(go.Scatter(
        x=hist['date'], 
        y=hist['price_eur_mwh'], 
        name="Cena rzeczywista (TGE)", 
        line=dict(color='#1f77b4', width=2)
    ))

    if not df_back.empty:
        back = df_back[df_back['date'] >= start_view]
        fig.add_trace(go.Scatter(
            x=back['date'], 
            y=back['predicted_price'], 
            name="Przewidywania / Prognoza", 
            line=dict(color='#d62728', width=2, dash='dot')
        ))

    fig.add_vline(x=now_utc, line_width=2, line_dash="solid", line_color="#2ca02c")
    fig.add_annotation(x=now_utc, text="TERAZ", showarrow=False, yshift=10, font=dict(color="#2ca02c"))

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=700,
        xaxis_title="Data",
        yaxis_title="EUR/MWh",
        xaxis=dict(autorange=True) 
    )

    st.plotly_chart(fig, use_container_width=True)

    if not df_back.empty:
        st.divider()
        merged = pd.merge(hist[['date', 'price_eur_mwh']], back[['date', 'predicted_price']], on='date').dropna()
        
        if not merged.empty:
            mae = (merged['price_eur_mwh'] - merged['predicted_price']).abs().mean()
            st.metric("Średni błąd modelu (MAE) na historii", f"{mae:.2f} EUR/MWh")
            
            st.info(f"Ostatnia aktualizacja ceny rynkowej: {hist['date'].max().strftime('%Y-%m-%d %H:%M')}. "
                    f"Prognoza wybiega do: {back['date'].max().strftime('%Y-%m-%d %H:%M')}")

except Exception as e:
    st.error(f"Błąd: {e}")