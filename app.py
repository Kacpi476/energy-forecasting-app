import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path

# Konfiguracja strony
st.set_page_config(
    page_title="Prognozowanie Cen Energii | Licencjat", 
    layout="wide",
    page_icon="⚡"
)

# --- FUNKCJE POMOCNICZE ---

@st.cache_data
def load_data():
    """Wczytuje dane historyczne i najnowsze prognozy."""
    df_hist = pd.read_parquet("data/final_training_data.parquet")
    df_forecast = pd.read_json("data/latest_forecast.json")
    
    # Konwersja dat
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    df_forecast['date'] = pd.to_datetime(df_forecast['date'])
    
    return df_hist, df_forecast

def get_feature_importance():
    """Wyciąga ważność cech z zapisanego modelu Random Forest."""
    try:
        model = joblib.load("machine_learning/model.joblib")
        # Lista cech musi zgadzać się z tą użytą w train_model.py
        features = ['demand', 'pv', 'wi', 'co2_price_eur', 'hour', 'day_of_week', 'res_share', 'price_lag_1']
        importance = pd.DataFrame({
            'Cecha': features,
            'Wpływ [%]': model.feature_importances_ * 100
        }).sort_values(by='Wpływ [%]', ascending=True)
        return importance
    except:
        return None

# --- GŁÓWNA LOGIKA APLIKACJI ---

try:
    df_hist, df_forecast = load_data()
    importance_df = get_feature_importance()

    # Nagłówek
    st.title("⚡ Inteligentny System Prognozowania Cen Energii")
    st.markdown(f"**Status systemu:** Operacyjny | **Ostatnia prognoza:** {df_forecast['forecast_made_at'].iloc[0].strftime('%Y-%m-%d %H:%M')}")
    st.divider()

    # Zakładki
    tab1, tab2, tab3 = st.tabs(["🔮 Prognoza Bieżąca", "📊 Analityka Modelu", "📑 Dane Surowe"])

    with tab1:
        # Metryki
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            last_price = df_hist['price_eur_mwh'].iloc[-1]
            st.metric("Ostatnia cena (TGE)", f"{last_price:.2f} EUR")
        with m2:
            avg_forecast = df_forecast['predicted_price'].mean()
            st.metric("Średnia prognoza 24h", f"{avg_forecast:.2f} EUR")
        with m3:
            co2_val = df_hist['co2_price_eur'].iloc[-1]
            st.metric("Cena CO2 (Proxy)", f"{co2_val:.2f} USD")
        with m4:
            res_val = (df_hist['res_share'].iloc[-1] * 100)
            st.metric("Udział OZE", f"{res_val:.1f}%")

        # Wykres główny
        st.subheader("Prognoza cen na najbliższe godziny")
        fig = go.Figure()

        # Dane historyczne (ostatnie 72h dla kontekstu)
        hist_view = df_hist.tail(72)
        fig.add_trace(go.Scatter(
            x=hist_view['date'], y=hist_view['price_eur_mwh'],
            name="Cena rzeczywista",
            line=dict(color='#1f77b4', width=3)
        ))

        # Prognoza
        fig.add_trace(go.Scatter(
            x=df_forecast['date'], y=df_forecast['predicted_price'],
            name="Prognoza modelu RF",
            line=dict(color='#d62728', width=3, dash='dash')
        ))

        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabela pod wykresem
        st.write("📋 **Szczegółowe wartości prognozowane:**")
        st.dataframe(df_forecast[['date', 'predicted_price']].rename(columns={'date': 'Data', 'predicted_price': 'Prognoza [EUR]'}).head(12), use_container_width=True)

    with tab2:
        st.header("Interpretacja i Ewaluacja Modelu")
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("Ważność cech (Feature Importance)")
            if importance_df is not None:
                fig_imp = px.bar(
                    importance_df, x='Wpływ [%]', y='Cecha', 
                    orientation='h', color='Wpływ [%]',
                    color_continuous_scale='Reds'
                )
                fig_imp.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("Nie można załadować modelu do analizy cech.")

        with c2:
            st.subheader("Informacje o modelu")
            st.info("""
            **Metodologia:**
            - **Model:** Random Forest Regressor (100 drzew).
            - **Zmienne:** Popyt krajowy, generacja wiatrowa i słoneczna, kurs KEUA, opóźnienie cenowe (lag-1).
            - **Horyzont:** 24 godziny.
            
            **Zastosowanie:**
            Model służy do estymacji trendów na Rynku Dnia Następnego (RDN). Największy wpływ na błąd mają nagłe zmiany pogodowe nieujęte w prognozie krótkoterminowej.
            """)

    with tab3:
        st.header("Podgląd zbioru treningowego")
        st.write("Ostatnie 100 rekordów z `final_training_data.parquet`:")
        st.dataframe(df_hist.tail(100), use_container_width=True)

except Exception as e:
    st.error("Błąd wczytywania danych. Upewnij się, że pliki .parquet i .json istnieją w folderze data.")
    st.exception(e)