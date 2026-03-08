import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Validación Econométrica - Remesas", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    
    [data-testid="stSidebar"] { background-color: #00285E; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    .sidebar-title { color: white !important; font-size: 24px; font-weight: bold; margin-bottom: 20px; }

    .stButton>button {
        color: white;
        background-color: #0B005E;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }

    .model-buttons button:hover {
        background-color: #e0e0e0 !important; /* hover gris claro */
}        

    .metric-container {
        background: linear-gradient(135deg, #004A99 0%, #002D62 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 2px solid #7FDBFF;
    }
    .metric-value { font-size: 22px; font-weight: bold; color: #7FDBFF; }

    .custom-table-container { display: flex; width: 100%; border-radius: 10px; overflow: hidden; }
    .table-col { flex: 1; display: flex; flex-direction: column; }
    .header-blue { background-color: #001f3f; color: white; padding: 12px; text-align: center; font-weight: bold; border: 0.5px solid #002D62; }
    .cell-blue { background-color: #004A99; color: white; padding: 10px; text-align: center; border: 0.5px solid #002D62; }

    .context-box { 
        background-color: #E3F2FD; 
        padding: 25px; 
        border-radius: 10px; 
        border-left: 8px solid #00285E; 
        color: #01579B; 
        text-align: justify;
        margin-bottom: 25px;
    }

    .sarima-box { 
        background-color: #E3F2FD; 
        padding: 25px; 
        border-radius: 10px; 
        border-left: 8px solid #00285E; 
        color: #1A237E; 
        text-align: justify;
        margin-bottom: 25px;
    }
    
    .interpretation-card { 
        background-color: #F8FAFC; 
        padding: 18px; 
        border-radius: 10px; 
        border: 1px solid #DEE2E6; 
        color: #334155;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- CARGA DE DATOS ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=';', encoding='utf-8')
    df['Fecha'] = pd.to_datetime(df['Ano'].astype(str) + '-' + df['Num_Mes'].astype(str) + '-01')
    df = df.sort_values('Fecha').reset_index(drop=True)
    df['Mes_Nombre'] = df['Mes'].str.strip()
    return df

try:
    df_2026 = load_data('Remesas2002_2026.csv')
    df_2024 = load_data('Remesas2002_2024.csv')
except Exception as e:
    st.error(f"Error: {e}"); st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="sidebar-title">Filtros</p>', unsafe_allow_html=True)
    
    # Slider de años
    y_range = st.slider("Comportamiento Histórico - Años", 2002, 2026, (2002, 2026))
    
    st.write("#### Selección de Meses")
    lista_meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", 
                   "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

    # Inicializar sesión si no existe
    if 'ms' not in st.session_state:
        st.session_state.ms = lista_meses.copy()

    # Botones para seleccionar/deseleccionar todos
    c_all, c_none = st.columns(2)
    if c_all.button("Seleccionar Todo"):
        st.session_state.ms = lista_meses.copy()
    if c_none.button("Deseleccionar Todo"):
        st.session_state.ms = []

    # Crear checkboxes para cada mes
    m_final = [m for m in lista_meses if st.checkbox(m, value=(m in st.session_state.ms))]
    st.session_state.ms = m_final

    # Mensaje cuando no hay selección
    if not m_final:
        st.info("Selecciona un mes para continuar")

    st.markdown("<br><br>" * 4, unsafe_allow_html=True)
    st.markdown("---")
    
    st.write("**Autores**")
    st.markdown("""
    1. Lilian María Elías Reyes
    2. Luis Roberto Ramírez Alvarado  
    3. Leslie Giselle Juárez Tobar 
    4. Jose Rafael Ignacio Vera Figueroa
    """)

# --- TÍTULO Y CONTEXTO ---
st.markdown(
    """
    <div style="text-align: center;">
        <h1>Del Dato al Futuro: Validación econométrica de metodologías para pronosticar remesas en Guatemala</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div class="context-box">
            <div style="text-align:center;">
    <b>Contexto del Proyecto:</b><br>
            </div>
    Las remesas familiares se han convertido en un pilar de la economía guatemalteca, representando una fuente clave de divisas y de ingreso para millones de hogares. 
    Su comportamiento refleja tanto las dinámicas migratorias como factores externos, por lo que es importante analizar su evolución en el tiempo. Este estudio analiza la serie de remesas utilizando datos de 2002 a 2024 para estimar modelos econométricos y una extensión hasta 2026 para evaluar pronósticos, 
comparando distintos modelos mediante el RMSE y proyectando valores de febrero 2026 a enero 2027.
</div>
""", unsafe_allow_html=True)

# --- GRÁFICO 1 ---
st.markdown(
    f"<h2 style='text-align: center;'>Comportamiento Histórico de Remesas {y_range[0]}–{y_range[1]}</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<h5 style='text-align: center;'>En Millones de USD</h5>",
    unsafe_allow_html=True
)

df_h = df_2026[(df_2026['Ano'] >= y_range[0]) & (df_2026['Ano'] <= y_range[1]) & (df_2026['Mes_Nombre'].isin(st.session_state.ms))]
x_t = np.arange(len(df_h)).reshape(-1, 1)
reg_h = LinearRegression().fit(x_t, df_h['Divisas'])
fig1 = go.Figure()
# CAMBIO: Color gris pálido y línea punteada para datos reales
fig1.add_trace(go.Scatter(x=df_h['Fecha'], y=df_h['Divisas'], name="Divisas Reales", line=dict(color='#D3D3D3', dash='dot', width=2)))
fig1.add_trace(go.Scatter(x=df_h['Fecha'], y=reg_h.predict(x_t), name="Tendencia", line=dict(color='#FFE100', width=2.5)))
fig1.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Año", yaxis_title="Millones de USD", height=700)
st.plotly_chart(fig1, use_container_width=True)

# --- SECCIÓN SARIMA ---
st.markdown("""
<div class="sarima-box">
            <div style="text-align:center;">
    <b>MODELO SARIMA:</b><br> 
    <b>Seasonal AutoRegressive Integrated Moving Average:</b><br>  
</div>
      
<b> - ¿Qué es?:</b> Es un modelo estadístico avanzado que captura tanto la tendencia como la estacionalidad de una serie de tiempo.<br>
<b> - ¿Cómo se aplica?:</b> Se identifican los parámetros de autorregresión, integración y media móvil, aplicándolos tanto a los datos inmediatos como a los ciclos estacionales (en este caso, 12 meses).<br>
<b> - Ventajas:</b> Alta precisión y robustez ante variaciones cíclicas complejas.<br>
<b> - Desventajas:</b> Requiere una configuración técnica exhaustiva y mayor capacidad computacional que los modelos simples.<br>
<b> - Comportamiento:</b> Ha demostrado ser el modelo con mayor fidelidad en este estudio, adaptándose mejor que cualquier otro a los choques estacionales históricos de la economía guatemalteca.

</div>
""", unsafe_allow_html=True)

# --- TABLA Y GRÁFICA LADO A LADO DE SARIMA ---
col_tabla, col_grafica = st.columns([1,2])

# TABLA
with col_tabla:
    st.markdown("""
<div style="display:flex; justify-content:center; margin-top:25px;">

<div style="
background-color:var(--background-color);
padding:25px;
border-radius:10px;
box-shadow:0px 2px 8px rgba(0,0,0,0.2);
width:100%;
color:var(--text-color);
border:1px solid rgba(128,128,128,0.2);
">

<h4 style="text-align:center; margin-bottom:0px;">Pronóstico del Flujo de Remesas de Abril 2024 a Marzo 2025</h4>
<p style="text-align:center; opacity:0.7; margin-top:3px;">
<i>Modelo SARIMA</i>
</p>

<table style="width:100%; border-collapse: collapse; font-size:15px; text-align:center;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:10px;">Mes-año</th>
<th style="padding:10px;">Flujo de Remesas (millones USD)</th>
</tr>
</thead>

<tbody>
<tr><td>abr-24</td><td>1,725.19</td></tr>
<tr><td>may-24</td><td>1,868.50</td></tr>
<tr><td>jun-24</td><td>1,887.65</td></tr>
<tr><td>jul-24</td><td>1,840.73</td></tr>
<tr><td>ago-24</td><td>1,902.59</td></tr>
<tr><td>sept-24</td><td>1,799.99</td></tr>
<tr><td>oct-24</td><td>1,879.96</td></tr>
<tr><td>nov-24</td><td>1,731.84</td></tr>
<tr><td>dic-24</td><td>1,905.96</td></tr>
<tr><td>ene-25</td><td>1,619.98</td></tr>
<tr><td>feb-25</td><td>1,674.07</td></tr>
<tr><td>mar-25</td><td>1,930.56</td></tr>
</tbody>
</table>

</div>
</div>
""", unsafe_allow_html=True)


# GRÁFICA SARIMA
with col_grafica:

    forecast_valores = [
        1725.19, 1868.50, 1887.65, 1840.73,
        1902.59, 1799.99, 1879.96, 1731.84,
        1905.96, 1619.98, 1674.07, 1930.56
    ]

    fechas_futuras = pd.date_range(
        start="2024-04-01",
        periods=12,
        freq="MS"
    )

    df_forecast = pd.DataFrame({
        "Fecha": fechas_futuras,
        "Divisas": forecast_valores
    })

    fig = go.Figure()

    # histórico
    fig.add_trace(go.Scatter(
        x=df_2024["Fecha"],
        y=df_2024["Divisas"],
        mode="lines",
        name="Histórico",
        line=dict(dash= "dot", color="#D3D3D3"),
    ))

    # pronóstico
    fig.add_trace(go.Scatter(
        x=df_forecast["Fecha"],
        y=df_forecast["Divisas"],
        mode="lines+markers",
        name="Pronóstico 2024-2025",
        line=dict(color="#2EC7FF"),

    ))

    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Millones de USD",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)


# --- SECCIÓN 2: MODELOS (INTERPRETACIÓN INCLUIDA) ---
st.markdown("---")
st.markdown(
    "<h2 style='text-align: center;'>Evaluación de Modelos Simples para el pronóstico de Abril 2024 a Marzo 2025</h2>",
    unsafe_allow_html=True
)

# Columnas para botones
sp1, col_b1, col_b2, col_b3, col_b4, col_b5, right = st.columns([1,1,1,1,1,1,1])

# Inicializar selección de modelo
if 'sel_mod' not in st.session_state:
    st.session_state.sel_mod = "HoltWinters"

# Botones
if col_b1.button("Datos Originales"):
    st.session_state.sel_mod = "Originales"
if col_b2.button("Promedios Móviles"):
    st.session_state.sel_mod = "PM"
if col_b3.button("Holt-Winters"):
    st.session_state.sel_mod = "HoltWinters"
if col_b4.button("Desestacionalización"):
    st.session_state.sel_mod = "Des"
if col_b5.button("Comparativa"):
    st.session_state.sel_mod = "Comparativa"

# Modelo seleccionado
mod = st.session_state.sel_mod

# Datos base
y24 = df_2024['Divisas'].values
f_dates_24 = pd.date_range(start='2024-04-01', periods=12, freq='MS')

# Variables interpretación vacías
v, d, c = "", "", ""

# --- MODELOS INDIVIDUALES ---
if mod == "PM":
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_2024['Fecha'], y=y24, name="Real", line=dict(color="#D3D3D3", width=1.5, dash='dot')))
    pm = pd.Series(y24).rolling(3).mean()
    pm_c = pm.rolling(2).mean().dropna()
    fig2.add_trace(go.Scatter(x=df_2024['Fecha'].iloc[-len(pm_c):], y=pm_c, name="Ajuste PM", line=dict(color="#00E878", width=2.5)))
    f_pm = [1261.43, 1266.38, 1271.33, 1276.29,
            1281.24, 1286.19, 1291.14, 1296.09,
            1301.05, 1306.00, 1310.95, 1315.90]
    fig2.add_trace(go.Scatter(x=f_dates_24, y=f_pm, name="Pronóstico PM", line=dict(color="#00733C", width=2)))
    v, d, c = (
        "Suaviza fluctuaciones aleatorias y el ruido de la serie.",
        "Presenta un rezago respecto a la tendencia actual.",
        "Regulariza los datos originales; las remesas siguen una tendencia al alza constante."
    )

elif mod == "HoltWinters":
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_2024['Fecha'], y=y24, name="Real", line=dict(color="#D3D3D3", width=1.5, dash='dot')))
    hw = ExponentialSmoothing(y24, trend="add", seasonal="multiplicative", seasonal_periods=12).fit()
    fig2.add_trace(go.Scatter(x=df_2024['Fecha'], y=hw.fittedvalues, name="Ajuste HW", line=dict(color="#D1C4E9", width=2)))
    f_hw = [1699.59, 1850.07, 1889.53, 1773.35, 1888.30, 1790.20, 1867.79, 1699.69, 1902.08, 1572.78, 1611.29, 1882.83]
    fig2.add_trace(go.Scatter(x=f_dates_24, y=f_hw, name="HW Pronóstico", line=dict(color="#AA00FF", width=3.5)))
    v, d, c = (
        "Modelado integral de tendencia y estacionalidad cambiante.",
        "Alta sensibilidad a los parámetros iniciales.",
        "Sigue la tendencia positiva y representa la estacionalidad en la proyección."
    )

elif mod == "Des":
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_2024['Fecha'], y=y24, name="Real", line=dict(color="#D3D3D3", width=1.5, dash='dot')))
    y_ts = pd.Series(y24, index=df_2024['Fecha'])
    decomposition = seasonal_decompose(y_ts, model='multiplicative', period=12)
    y_des = y24 / decomposition.seasonal.values
    fig2.add_trace(go.Scatter(x=df_2024['Fecha'], y=y_des, name="Desestacionalizada", line=dict(color="#FF4081", width=2)))
    f_des = [1103.59, 1112.43, 1316.00, 1298.88,
             1395.23, 1363.98, 1348.66, 1379.16,
             1295.11, 1381.73, 1224.59, 1342.97]
    fig2.add_trace(go.Scatter(x=f_dates_24, y=f_des, name="Pronóstico Desestacionalizado", line=dict(color="#C90043", width=3)))
    v, d, c = (
        "Identifica crecimiento subyacente sin ruido mensual.",
        "No es un modelo predictivo, es una técnica de análisis.",
        "La proyección refleja meses fuertes y débiles coherentes con el comportamiento económico."
    )

elif mod == "Originales":
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_2024['Fecha'], y=y24, name="Real", line=dict(color="#D3D3D3", width=1.5, dash='dot')))
    reg = LinearRegression().fit(np.arange(len(y24)).reshape(-1,1), y24)
    f_orig = reg.predict(np.arange(len(y24), len(y24)+12).reshape(-1,1))
    fig2.add_trace(go.Scatter(
    x=np.concatenate([df_2024['Fecha'], f_dates_24]), 
    y=np.concatenate([reg.predict(np.arange(len(y24)).reshape(-1,1)), f_orig]),
    name="Tendencia",
    line=dict(color="#FFAB40", width=2)
))
    
    # Marcadores destacados para los 12 meses futuros
    fig2.add_trace(go.Scatter(
        x=f_dates_24,
        y=f_orig,
        name="Pronóstico Futuros",
        mode="markers",
        marker=dict(color="#FF8400", size=8, symbol="circle")
    ))

    v, d, c = (
        "Simplicidad y fácil interpretación.",
        "Ignora ciclos estacionales.",
        "Trajectoria de remesas con incremento mensual constante."
    )

# --- COMPARATIVA ---
elif mod == "Comparativa":
    f_sarima = [1725.19, 1868.5, 1887.65, 1840.73, 1902.59, 1799.99, 1879.96, 1731.84, 1905.96, 1619.98, 1674.07, 1930.56]
    f_originales = [1265.31, 1270.29, 1275.27, 1280.24, 1285.22, 1290.20, 1295.18, 1300.16, 1305.14, 1310.11, 1315.09, 1320.07]
    f_pm = [1261.43, 1266.38, 1271.33, 1276.29, 1281.24, 1286.19, 1291.14, 1296.09, 1301.05, 1306.00, 1310.95, 1315.90]
    f_hw = [1699.59, 1850.07, 1889.53, 1773.35, 1888.30, 1790.20, 1867.79, 1699.69, 1902.08, 1572.78, 1611.29, 1882.83]
    f_des = [1103.59, 1112.43, 1316.00, 1298.88, 1395.23, 1363.98, 1348.66, 1379.16, 1295.11, 1381.73, 1224.59, 1342.97]

    pronosticos = {"SARIMA": f_sarima, "Datos Originales": f_originales, "Promedios Móviles": f_pm, "Holt-Winters": f_hw, "Desestacionalización": f_des}
    colores = {"SARIMA": "#00285E", "Datos Originales": "#FF8400", "Promedios Móviles": "#00733C", "Holt-Winters": "#AA00FF", "Desestacionalización": "#C90043"}
    estilos = {"SARIMA": "solid", "Datos Originales": "solid", "Promedios Móviles": "dash", "Holt-Winters": "solid", "Desestacionalización": "dash"}

    fig2 = go.Figure()
    for nombre, valores in pronosticos.items():
        fig2.add_trace(go.Scatter(
            x=f_dates_24,
            y=valores,
            name=nombre,
            line=dict(color=colores[nombre], width=3, dash=estilos[nombre]),
            mode="lines+markers",
            hovertemplate="%{y:.2f} millones USD<br>%{x|%b %Y}<extra>" + nombre + "</extra>"
        ))
    fig2.update_layout(
        template="plotly_dark",
        title=dict(text="Comparativa de Pronósticos 2024-2025", x=0.5, xanchor='center', font=dict(size=22)),
        xaxis_title="Mes",
        yaxis_title="Remesas (millones USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title="Método", bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hovermode="x unified",
        height=650
    )

# --- Mostrar la figura ---
fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

# --- Cuadros de interpretación ---
if mod != "Comparativa":
    it1, it2, it3 = st.columns(3)
    with it1: st.markdown(f'<div class="interpretation-card"><b>Ventaja:</b><br>{v}</div>', unsafe_allow_html=True)
    with it2: st.markdown(f'<div class="interpretation-card"><b>Desventaja:</b><br>{d}</div>', unsafe_allow_html=True)
    with it3: st.markdown(f'<div class="interpretation-card"><b>Comportamiento:</b><br>{c}</div>', unsafe_allow_html=True)


# --- RMSE ---
st.markdown("---")
st.markdown(
    "<h3 style='text-align:center; margin-top:20px;'>Comparativa de RMSE (Root Mean Square Error)</h3>",
    unsafe_allow_html=True
)
rm_cols = st.columns(5)
metrics = [("SARIMA", "110.1685"), ("Originales", "589.4224"), ("P. Móviles", "593.3804"), ("Holt-Winters", "134.9173"), ("Desestac.", "594.86")]
for i, (m, v) in enumerate(metrics):
    rm_cols[i].markdown(f'<div class="metric-container"><b>{m}</b><br><span class="metric-value">{v}</span></div>', unsafe_allow_html=True)



# --- SECCIÓN 3: PRONÓSTICO FINAL ---
st.markdown("---")
st.markdown(
    "<h2 style='text-align:center;'>Pronóstico Final de Febrero 2026 a Enero 2027 - Modelo Holt-Winters</h2>",
    unsafe_allow_html=True
)

y_f = df_2026['Divisas'].values
model_f = ExponentialSmoothing(y_f, trend="add", seasonal="multiplicative", seasonal_periods=12).fit()
forecast_f = model_f.forecast(12)
rmse_f = np.sqrt(mean_squared_error(y_f, model_f.fittedvalues))
f_idx = pd.date_range(start='2026-02-01', periods=12, freq='MS')

# --- Columnas: Tabla a la izquierda, Gráfica a la derecha ---
col_tab, col_graf = st.columns([1, 2])

# --- Columna izquierda: Tabla con estilo SARIMA ---
with col_tab:
    st.markdown("""
<div style="display:flex; justify-content:center; margin-top:25px;">

<div style="
background-color:var(--background-color);
padding:25px;
border-radius:10px;
box-shadow:0px 2px 8px rgba(0,0,0,0.2);
width:100%;
color:var(--text-color);
border:1px solid rgba(128,128,128,0.2);
">

<h4 style="text-align:center; margin-bottom:0px;">Pronóstico del Flujo de Remesas de Febrero 2026 a Enero 2027</h4>
<p style="text-align:center; opacity:0.7; margin-top:3px;">
<i>Modelo Holt-Winters</i>
</p>

<table style="width:100%; border-collapse: collapse; font-size:15px; text-align:center;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:10px;">Mes-año</th>
<th style="padding:10px;">Flujo de Remesas (millones USD)</th>
</tr>
</thead>
            <tbody>
                <tr><td>feb-26</td><td>1,973.30</td></tr>
                <tr><td>mar-26</td><td>2,274.13</td></tr>
                <tr><td>abr-26</td><td>2,266.25</td></tr>
                <tr><td>may-26</td><td>2,490.85</td></tr>
                <tr><td>jun-26</td><td>2,454.75</td></tr>
                <tr><td>jul-26</td><td>2,411.17</td></tr>
                <tr><td>ago-26</td><td>2,545.68</td></tr>
                <tr><td>sep-26</td><td>2,332.44</td></tr>
                <tr><td>dic-26</td><td>2,525.23</td></tr>
                <tr><td>ene-27</td><td>2,212.17</td></tr>
                <tr><td>feb-27</td><td>2,495.96</td></tr>
                <tr><td>mar-27</td><td>2,130.71</td></tr>
</tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Columna derecha: Gráfica ---
with col_graf:
    fig3 = go.Figure()
    # Histórico en naranja/dorado
    fig3.add_trace(go.Scatter(x=df_2026['Fecha'], y=df_2026['Divisas'], 
                              name="Histórico Real", line=dict(color='#D3D3D3', width=2, dash='dot')))
    # Pronóstico en verde limón
    fig3.add_trace(go.Scatter(x=f_idx, y=forecast_f, 
                              name="Pronóstico Feb 26 - Ene 27", line=dict(color="#b5ff3c", width=4)))
    # Intervalo de confianza
    fig3.add_trace(go.Scatter(
        x=f_idx.tolist() + f_idx.tolist()[::-1],
        y=(forecast_f + 1.96*rmse_f).tolist() + (forecast_f - 1.96*rmse_f).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(181, 255, 60, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name="Confianza 95%"
    ))
    fig3.update_layout(template="plotly_dark", height=700,
                       xaxis_title="Fecha", yaxis_title="Millones USD")
    st.plotly_chart(fig3, use_container_width=True)

st.write("")
# --- Justificación debajo ---
st.markdown("""
<div style="background-color: #E1F5FE; 
            padding: 25px; 
            border-radius: 10px; 
            border-left: 8px solid #004A99; 
            color: #01579B; 
            text-align: justify;">
    <b style="font-size: 18px;">Justificación Técnica del Modelo Seleccionado:</b><br><br>
    Se ha optado por el modelo <b>Holt-Winters con Estacionalidad Multiplicativa</b> debido a su alta sensibilidad ante los patrones 
    cíclicos que presentan las remesas en Guatemala. A diferencia de un modelo aditivo, el enfoque multiplicativo reconoce que los 
    picos estacionales (como los observados en mayo y diciembre) tienden a crecer en magnitud a medida que la tendencia general 
    de la serie aumenta.<br><br>
    Esta validación asegura que el pronóstico no sea solo una proyección lineal inerte, sino que responda a la dinámica real del 
    flujo de divisas, manteniendo un equilibrio óptimo entre la captura de la tendencia a largo plazo y la volatilidad estacional.<br><br>
    El mejor modelo básico para pronosticar las remesas en 2026 sería <b>Holt Winters</b>, porque combina nivel, tendencia y estacionalidad en una sola estructura suavizada. Los resultados muestran que obtuvo el RMSE más bajo, lo cual indica mayor precisión predictiva. Es un método transparente, fácil de actualizar y suficientemente robusto para un banco central que necesita proyecciones mensuales consistentes sin requerir modelos complejos o variables externas.
</div>
""", unsafe_allow_html=True)
