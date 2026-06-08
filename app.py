import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    layout="wide",
    page_title="Monitoring Cen Energii – Praca Licencjacka",
    page_icon="⚡"
)

# ── Globalne style
st.markdown("""
<style>
    .section-header {
        background: linear-gradient(90deg, #1a2744 0%, #0e1628 100%);
        border-left: 4px solid #4fc3f7;
        padding: 12px 20px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .section-header h3 { color: #e8f4fd; margin: 0 0 4px 0; font-size: 1.1rem; }
    .section-header p  { color: #90b8d4; margin: 0; font-size: 0.85rem; }
    .metric-card {
        background: #0e1628;
        border: 1px solid #1e3050;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
    .metric-card .lbl { font-size: 0.8rem; color: #90b8d4; margin-top: 4px; }
    .info-box {
        background: #0d1f35;
        border: 1px solid #1e3050;
        border-left: 3px solid #4fc3f7;
        border-radius: 4px;
        padding: 10px 16px;
        color: #90b8d4;
        font-size: 0.83rem;
        margin-bottom: 12px;
    }
    .warn-box {
        background: #2d1a00;
        border-left: 3px solid #ff9800;
        border-radius: 4px;
        padding: 10px 16px;
        color: #ffcc80;
        font-size: 0.83rem;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Ładowanie danych
@st.cache_data(ttl=300)
def load_data():
    errors = []

    try:
        df = pd.read_parquet("data/final_training_data.parquet")
        df['date'] = pd.to_datetime(df['date'], utc=True)
    except FileNotFoundError:
        df = pd.DataFrame()
        errors.append("Brak pliku `data/final_training_data.parquet` – dane historyczne niedostępne.")
    except Exception as e:
        df = pd.DataFrame()
        errors.append(f"Błąd odczytu danych historycznych: {e}")

    try:
        df_back = pd.read_parquet("data/forecast_history.parquet")
        df_back['date'] = pd.to_datetime(df_back['date'], utc=True)
    except FileNotFoundError:
        df_back = pd.DataFrame()
        errors.append("Brak pliku `data/forecast_history.parquet` – dane prognoz niedostępne.")
    except Exception as e:
        df_back = pd.DataFrame()
        errors.append(f"Błąd odczytu prognoz: {e}")

    return df, df_back, errors

df, df_back, load_errors = load_data()

# ── Nagłówek strony
st.title("⚡ Monitoring systemu predykcji cen energii elektrycznej")
st.caption(
    "Praca licencjacka – Kacper Knapczyk | Uniwersytet Ekonomiczny w Krakowie, 2026  "
    "| Model: Random Forest · Dane: ENTSO-E, PSE, Open-Meteo"
)
st.divider()

for err in load_errors:
    st.markdown(f'<div class="warn-box">{err}</div>', unsafe_allow_html=True)

if df.empty and df_back.empty:
    st.error("Brak jakichkolwiek danych do wyświetlenia. Sprawdź katalog `data/`.")
    st.stop()

# ── Stałe 
START_VIEW   = pd.Timestamp("2026-01-01", tz='UTC')
NOW_UTC      = pd.Timestamp("2026-06-06 22:00:00", tz='UTC')  # DEMO: linia "TERAZ" na potrzeby pracy licencjackiej
DARK_TEMPLATE = "plotly_dark"

# Budujemy hist z dwóch źródeł:
# 1. final_training_data.parquet – historyczne ceny treningowe
# 2. forecast_history.parquet    – ceny realne dodane przez add_real_prices.py
#    (dla godzin po zakończeniu zbioru treningowego)
if not df.empty:
    hist_train = df[['date', 'price_eur_mwh']].copy()
else:
    hist_train = pd.DataFrame(columns=['date', 'price_eur_mwh'])

if not df_back.empty and 'price_eur_mwh' in df_back.columns:
    # Weź tylko te rekordy z forecast_history gdzie jest cena realna
    hist_back = df_back[df_back['price_eur_mwh'].notna()][['date', 'price_eur_mwh']].copy()
else:
    hist_back = pd.DataFrame(columns=['date', 'price_eur_mwh'])

# Połącz oba źródła, usuń duplikaty (pierwszeństwo ma forecast_history dla nakładających się dat)
hist_combined = pd.concat([hist_train, hist_back], ignore_index=True)
hist_combined = hist_combined.sort_values('date')
hist_combined = hist_combined.drop_duplicates(subset='date', keep='last')

hist = hist_combined[hist_combined['date'] >= START_VIEW].dropna(subset=['price_eur_mwh'])

back = df_back[df_back['date'] >= START_VIEW] if not df_back.empty else pd.DataFrame()

if not back.empty:
    # Ostatnia znana cena rzeczywista — tu kończy się niebieska linia
    last_real_ts = hist['date'].max() if not hist.empty else NOW_UTC

    # back_hist   — prognoza historyczna (do porównania z ceną realną)
    # back_bridge — strefa między końcem cen realnych a TERAZ (wypełnia przerwę)
    # back_future — prognoza operacyjna (po linii TERAZ)
    back_hist   = back[back['date'] <= last_real_ts]
    back_bridge = back[(back['date'] > last_real_ts) & (back['date'] <= NOW_UTC)]
    back_future = back[back['date'] >  NOW_UTC]
else:
    back_hist   = pd.DataFrame()
    back_bridge = pd.DataFrame()
    back_future = pd.DataFrame()

# SEKCJA 1 – Cena realna vs prognoza

st.markdown("""
<div class="section-header">
  <h3>1 · Cena realna TGE vs przewidywania modelu</h3>
  <p>Porównanie rzeczywistych notowań z rynku dnia następnego (ENTSO-E/TGE) z wyjściem modelu Random Forest.
     Zielona linia pionowa oznacza bieżący moment – po jej prawej stronie widoczna jest prognoza operacyjna.</p>
</div>
""", unsafe_allow_html=True)

if hist.empty:
    st.markdown('<div class="warn-box">Brak danych historycznych dla wykresu cen.</div>', unsafe_allow_html=True)
else:
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=hist['date'], y=hist['price_eur_mwh'],
        name="Cena rzeczywista (TGE)",
        line=dict(color='#4fc3f7', width=2),
        hovertemplate="%{x|%d.%m %H:%M}<br><b>%{y:.1f} EUR/MWh</b><extra>Rzeczywista</extra>"
    ))

    if not back_hist.empty:
        fig1.add_trace(go.Scatter(
            x=back_hist['date'], y=back_hist['predicted_price'],
            name="Backtest modelu",
            line=dict(color='#ff7043', width=1.5, dash='dot'),
            hovertemplate="%{x|%d.%m %H:%M}<br><b>%{y:.1f} EUR/MWh</b><extra>Backtest</extra>"
        ))

    # Bridge: łączy koniec cen realnych z linią TERAZ — eliminuje przerwę na wykresie
    # Powstaje gdy ceny realne są opóźnione względem czasu rzeczywistego
    if not back_bridge.empty:
        if not back_hist.empty:
            bridge_data = pd.concat([back_hist.tail(1), back_bridge], ignore_index=True)
        else:
            bridge_data = back_bridge

        fig1.add_trace(go.Scatter(
            x=bridge_data['date'], y=bridge_data['predicted_price'],
            name="Prognoza bieżąca (oczekiwanie na ceny)",
            line=dict(color='#ffd54f', width=2, dash='dash'),
            hovertemplate="%{x|%d.%m %H:%M}<br><b>%{y:.1f} EUR/MWh</b><extra>Prognoza bieżąca</extra>"
        ))

    if not back_future.empty:
        fig1.add_trace(go.Scatter(
            x=back_future['date'], y=back_future['predicted_price'],
            name="Prognoza (kolejne doby)",
            line=dict(color='#ffd54f', width=2.5),
            hovertemplate="%{x|%d.%m %H:%M}<br><b>%{y:.1f} EUR/MWh</b><extra>Prognoza</extra>"
        ))

        fig1.add_vrect(
            x0=NOW_UTC, x1=back_future['date'].max(),
            fillcolor="rgba(255, 213, 79, 0.06)",
            layer="below", line_width=0,
            annotation_text="strefa prognozy", annotation_position="top right",
            annotation_font=dict(color="#ffd54f", size=11)
        )

    fig1.add_vline(x=NOW_UTC, line_width=2, line_dash="solid", line_color="#69f0ae")
    fig1.add_annotation(
        x=NOW_UTC, y=1, yref="paper", text="  TERAZ",
        showarrow=False, font=dict(color="#69f0ae", size=12),
        xanchor="left"
    )

    fig1.update_layout(
        template=DARK_TEMPLATE,
        hovermode="x unified",
        height=480,
        xaxis_title="Data",
        yaxis_title="EUR/MWh",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20, b=10),
    )
    st.plotly_chart(fig1, use_container_width=True)

    if not back_hist.empty:
        merged = pd.merge(
            hist[['date', 'price_eur_mwh']],
            back_hist[['date', 'predicted_price']],
            on='date'
        ).dropna()

        if not merged.empty:
            mae_global = (merged['price_eur_mwh'] - merged['predicted_price']).abs().mean()
            r2 = 1 - ((merged['price_eur_mwh'] - merged['predicted_price'])**2).sum() / \
                     ((merged['price_eur_mwh'] - merged['price_eur_mwh'].mean())**2).sum()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="val">{mae_global:.1f}</div>'
                            f'<div class="lbl">MAE globalny (EUR/MWh)</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="val">{r2:.3f}</div>'
                            f'<div class="lbl">R² (wsp. determinacji)</div></div>', unsafe_allow_html=True)
            with c3:
                last_real = hist['date'].max().strftime('%d.%m %H:%M')
                st.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.1rem">{last_real}</div>'
                            f'<div class="lbl">Ostatnia cena rzeczywista</div></div>', unsafe_allow_html=True)
            with c4:
                if not back_future.empty:
                    last_fc = back_future['date'].max().strftime('%d.%m %H:%M')
                else:
                    last_fc = "–"
                st.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.1rem">{last_fc}</div>'
                            f'<div class="lbl">Horyzont prognozy</div></div>', unsafe_allow_html=True)

st.divider()

# SEKCJA 2 – Udział OZE (res_share)
st.markdown("""
<div class="section-header">
  <h3>2 · Udział odnawialnych źródeł energii w miksie (res_share)</h3>
  <p>Zmienna res_share jest najważniejszym predyktorem modelu (waga Gini ≈ 0,60).
     Wykres ilustruje, jak chwilowy udział OZE wpływa na obniżenie ceny rynkowej – efekt kanibalizacji cen.</p>
</div>
""", unsafe_allow_html=True)

if df.empty or 'res_share' not in df.columns:
    st.markdown('<div class="warn-box">Kolumna <code>res_share</code> niedostępna w zbiorze danych.</div>',
                unsafe_allow_html=True)
else:
    res_data = df[df['date'] >= START_VIEW].dropna(subset=['res_share', 'price_eur_mwh'])

    if res_data.empty:
        st.markdown('<div class="warn-box">Brak danych OZE dla wybranego okresu.</div>', unsafe_allow_html=True)
    else:
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=res_data['date'],
            y=(res_data['res_share'] * 100).clip(0, 100),
            name="Udział OZE (%)",
            fill='tozeroy',
            line=dict(color='#66bb6a', width=1.5),
            fillcolor='rgba(102,187,106,0.15)',
            yaxis='y1',
            hovertemplate="%{x|%d.%m %H:%M}<br>OZE: <b>%{y:.1f}%</b><extra></extra>"
        ))

        fig2.add_trace(go.Scatter(
            x=res_data['date'],
            y=res_data['price_eur_mwh'],
            name="Cena energii (EUR/MWh)",
            line=dict(color='#4fc3f7', width=1.5, dash='dot'),
            yaxis='y2',
            hovertemplate="%{x|%d.%m %H:%M}<br>Cena: <b>%{y:.1f} EUR/MWh</b><extra></extra>"
        ))

        fig2.add_vline(x=NOW_UTC, line_width=2, line_dash="solid", line_color="#69f0ae")

        fig2.update_layout(
            template=DARK_TEMPLATE,
            hovermode="x unified",
            height=380,
            xaxis_title="Data",
            yaxis=dict(title="Udział OZE (%)", range=[0, 100], side='left'),
            yaxis2=dict(title="EUR/MWh", overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        corr_val = res_data['res_share'].corr(res_data['price_eur_mwh'])
        avg_res  = res_data['res_share'].mean() * 100
        st.markdown(
            f'<div class="info-box">📊 Korelacja Pearsona OZE↔cena: <b>{corr_val:.3f}</b> '
            f'&nbsp;|&nbsp; Średni udział OZE w analizowanym okresie: <b>{avg_res:.1f}%</b></div>',
            unsafe_allow_html=True
        )

st.divider()

# SEKCJA 3 – MAE tygodniowy
st.markdown("""
<div class="section-header">
  <h3>3 · Tygodniowy błąd predykcji (MAE)</h3>
  <p>Średni błąd bezwzględny agregowany w oknach 7-dniowych – pozwala ocenić stabilność modelu w czasie
     i wykryć okresy zwiększonej zmienności rynkowej, w których precyzja prognoz spada.</p>
</div>
""", unsafe_allow_html=True)

# Do MAE używamy pełnego back (zawiera predicted_price dla całej historii)
# back_hist byłby za wąski — jest cięty do last_real_ts
back_for_mae = back[back['predicted_price'].notna()] if not back.empty else pd.DataFrame()

if back_for_mae.empty or hist.empty:
    st.markdown('<div class="warn-box">Niewystarczające dane backtestu do obliczenia MAE tygodniowego.</div>',
                unsafe_allow_html=True)
else:
    merged_full = pd.merge(
        hist[['date', 'price_eur_mwh']],
        back_for_mae[['date', 'predicted_price']],
        on='date'
    ).dropna()

    if merged_full.empty:
        st.markdown('<div class="warn-box">Brak wspólnych rekordów dla obliczenia MAE.</div>',
                    unsafe_allow_html=True)
    else:
        merged_full = merged_full.set_index('date').sort_index()
        merged_full['abs_err'] = (merged_full['price_eur_mwh'] - merged_full['predicted_price']).abs()
        weekly_mae = merged_full['abs_err'].resample('7D').mean().dropna().reset_index()
        weekly_mae.columns = ['week', 'mae']

        if weekly_mae.empty:
            st.markdown('<div class="warn-box">Za mało danych do agregacji tygodniowej.</div>',
                        unsafe_allow_html=True)
        else:
            avg_mae = weekly_mae['mae'].mean()

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=weekly_mae['week'],
                y=weekly_mae['mae'],
                name="MAE tygodniowy",
                marker=dict(
                    color=weekly_mae['mae'],
                    colorscale=[[0, '#388e3c'], [0.5, '#f9a825'], [1, '#c62828']],
                    showscale=True,
                    colorbar=dict(title="EUR/MWh", thickness=12)
                ),
                hovertemplate="Tydzień od %{x|%d.%m.%Y}<br>MAE: <b>%{y:.2f} EUR/MWh</b><extra></extra>"
            ))
            fig3.add_hline(
                y=avg_mae, line_dash="dash", line_color="#90b8d4",
                annotation_text=f"  Średnia: {avg_mae:.1f} EUR/MWh",
                annotation_font=dict(color="#90b8d4", size=11)
            )
            fig3.update_layout(
                template=DARK_TEMPLATE,
                height=320,
                xaxis_title="Tydzień",
                yaxis_title="MAE (EUR/MWh)",
                showlegend=False,
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig3, use_container_width=True)

            best_week  = weekly_mae.loc[weekly_mae['mae'].idxmin()]
            worst_week = weekly_mae.loc[weekly_mae['mae'].idxmax()]
            st.markdown(
                f'<div class="info-box">'
                f'Najlepszy tydzień: <b>{best_week["week"].strftime("%d.%m.%Y")}</b> '
                f'– MAE = {best_week["mae"]:.2f} EUR/MWh &nbsp;|&nbsp; '
                f'Najtrudniejszy: <b>{worst_week["week"].strftime("%d.%m.%Y")}</b> '
                f'– MAE = {worst_week["mae"]:.2f} EUR/MWh'
                f'</div>',
                unsafe_allow_html=True
            )

st.divider()

# SEKCJA 4 – Tabela prognoz na najbliższe 24h
st.markdown("""
<div class="section-header">
  <h3>4 · Prognoza operacyjna – najbliższe 24 godziny</h3>
  <p>Wyjście modelu dla kolejnych godzin handlowych. Dane pogodowe i systemowe (Open-Meteo, PSE)
     są podstawą predykcji; horyzont prognozowania wynosi do 36h zgodnie z architekturą potoku danych.</p>
</div>
""", unsafe_allow_html=True)

if back_future.empty:
    st.markdown(
        '<div class="warn-box">Brak danych prognozy przyszłościowej. '
        'Uruchom <code>update_data.py</code>, aby wygenerować nowe predykcje.</div>',
        unsafe_allow_html=True
    )
else:
    next_24h = back_future[back_future['date'] <= NOW_UTC + pd.Timedelta(hours=24)].copy()

    if next_24h.empty:
        st.markdown('<div class="warn-box">Prognoza dostępna, ale poza oknem 24h. Sprawdź horyzont modelu.</div>',
                    unsafe_allow_html=True)
    else:
        next_24h = next_24h.dropna(subset=['predicted_price'])
        next_24h = next_24h.sort_values('date').reset_index(drop=True)

        display_cols = {'date': 'Data i godzina (UTC)', 'predicted_price': 'Prognoza ceny (EUR/MWh)'}
        if 'res_share' in next_24h.columns:
            display_cols['res_share'] = 'Udział OZE (%)'
        if 'temperature_c' in next_24h.columns:
            display_cols['temperature_c'] = 'Temp. (°C)'

        table = next_24h[list(display_cols.keys())].rename(columns=display_cols).copy()
        table['Data i godzina (UTC)'] = table['Data i godzina (UTC)'].dt.strftime('%d.%m.%Y %H:%M')
        table['Prognoza ceny (EUR/MWh)'] = table['Prognoza ceny (EUR/MWh)'].round(2)
        if 'Udział OZE (%)' in table.columns:
            table['Udział OZE (%)'] = (table['Udział OZE (%)'] * 100).round(1)
        if 'Temp. (°C)' in table.columns:
            table['Temp. (°C)'] = table['Temp. (°C)'].round(1)

        min_p = table['Prognoza ceny (EUR/MWh)'].min()
        max_p = table['Prognoza ceny (EUR/MWh)'].max()

        def color_price(val):
            try:
                if pd.isna(val) or max_p == min_p:
                    return 'background-color: #1a2744'
                t = (float(val) - min_p) / (max_p - min_p)
                r = int(46  + t * (198 - 46))
                g = int(142 + t * (40  - 142))
                b = int(60  + t * (40  - 60))
                return f'background-color: rgba({r},{g},{b},0.35)'
            except (TypeError, ValueError):
                return 'background-color: #1a2744'

        styled = table.style.applymap(color_price, subset=['Prognoza ceny (EUR/MWh)'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        avg_fc = next_24h['predicted_price'].mean()
        max_fc = next_24h['predicted_price'].max()
        min_fc = next_24h['predicted_price'].min()
        peak_h = next_24h.loc[next_24h['predicted_price'].idxmax(), 'date'].strftime('%H:%M')

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="val">{avg_fc:.1f}</div>'
                        f'<div class="lbl">Średnia prognozowana cena (EUR/MWh)</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="val">{max_fc:.1f}</div>'
                        f'<div class="lbl">Szczyt cenowy · godz. {peak_h} UTC</div></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="val">{min_fc:.1f}</div>'
                        f'<div class="lbl">Minimum cenowe w oknie 24h</div></div>',
                        unsafe_allow_html=True)

# ── Stopka
st.divider()
st.caption(
    f"Ostatnia aktualizacja widoku: {NOW_UTC.strftime('%Y-%m-%d %H:%M')} UTC  |  "
    "Model: Random Forest Regressor · n_estimators=200 · max_depth=12  |  "
    "Źródła: ENTSO-E, PSE, Open-Meteo, KEUA (CO₂)"
)

