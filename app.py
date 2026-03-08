import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Validación Econométrica - Remesas", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    
    [data-testid="stSidebar"] { background-color: #002D62; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    .sidebar-title { color: white !important; font-size: 24px; font-weight: bold; margin-bottom: 20px; }

    .stButton>button {
        color: black !important;
        background-color: #F0F2F6;
        border-radius: 5px;
        font-weight: bold;
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
        border-left: 8px solid #002D62; 
        color: #01579B; 
        text-align: justify;
        margin-bottom: 25px;
    }

    .sarima-box { 
        background-color: #E8EAF6; 
        padding: 25px; 
        border-radius: 10px; 
        border-left: 8px solid #3F51B5; 
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
    st.markdown('<p class="sidebar-title">Panel de Control</p>', unsafe_allow_html=True)
    y_range = st.slider("Rango de Años Histórico", 2002, 2026, (2002, 2026))
    
    st.write("### Selección de Meses")
    lista_meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    
    c_all, c_none = st.columns(2)
    if c_all.button("Seleccionar Todo"): st.session_state.ms = lista_meses.copy()
    if c_none.button("Deseleccionar Todo"): st.session_state.ms = []
    
    if 'ms' not in st.session_state: st.session_state.ms = lista_meses.copy()
    
    m_final = [m for m in lista_meses if st.checkbox(m, value=(m in st.session_state.ms))]
    st.session_state.ms = m_final

    st.markdown("<br><br>" * 4, unsafe_allow_html=True)
    st.markdown("---")
    st.write("**Autores**")
    st.markdown("""
    Lilian María Elías Reyes  
    -1246923  
    Luis Roberto Ramírez Alvarado  
    -1217523  
    Leslie Giselle Juárez Tobar  
    -1244823  
    Jose Rafael Ignacio Vera Figueroa  
    -1108523
    """)

# --- TÍTULO Y CONTEXTO ---
st.title("Del Dato al Futuro: Validación econométrica de metodologías para pronosticar remesas en Guatemala")

st.markdown("""
<div class="context-box">
    <b>Contexto del Proyecto:</b><br>
    Las remesas familiares se han convertido en un pilar de la economía guatemalteca, representando una fuente clave de divisas y de ingreso para millones de hogares. Su comportamiento refleja tanto las dinámicas 
    migratorias como factores externos, por lo que es importante analizar su evolución en el tiempo. Este estudio analiza la serie de remesas utilizando datos de 2002 a 2024 para estimar modelos econométricos y una 
    extensión hasta 2026 para evaluar pronósticos, comparando distintos modelos mediante el RMSE y proyectando valores de febrero 2026 a enero 2027.
</div>
""", unsafe_allow_html=True)

# --- GRÁFICO 1 ---
st.header("Comportamiento Histórico de Remesas 2002-2026")
df_h = df_2026[(df_2026['Ano'] >= y_range[0]) & (df_2026['Ano'] <= y_range[1]) & (df_2026['Mes_Nombre'].isin(st.session_state.ms))]
x_t = np.arange(len(df_h)).reshape(-1, 1)
reg_h = LinearRegression().fit(x_t, df_h['Divisas'])
fig1 = go.Figure()
# CAMBIO: Color gris pálido y línea punteada para datos reales
fig1.add_trace(go.Scatter(x=df_h['Fecha'], y=df_h['Divisas'], name="Divisas Reales", line=dict(color='#D3D3D3', width=2, dash='dot')))
fig1.add_trace(go.Scatter(x=df_h['Fecha'], y=reg_h.predict(x_t), name="Tendencia", line=dict(color='#FFD600')))
fig1.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig1, use_container_width=True)

# --- SECCIÓN SARIMA ---
st.markdown("""
<div class="sarima-box">
    <b>Modelo SARIMA (Seasonal AutoRegressive Integrated Moving Average):</b><br>
    <b>¿Qué es?:</b> Es un modelo estadístico avanzado que captura tanto la tendencia como la estacionalidad de una serie de tiempo.<br>
    <b>¿Cómo se aplica?:</b> Se identifican los parámetros de autorregresión, integración y media móvil, aplicándolos tanto a los datos inmediatos como a los ciclos estacionales (en este caso, 12 meses).<br>
    <b>Ventajas:</b> Alta precisión y robustez ante variaciones cíclicas complejas.<br>
    <b>Desventajas:</b> Requiere una configuración técnica exhaustiva y mayor capacidad computacional que los modelos simples.<br>
    <b>Comportamiento:</b> Ha demostrado ser el modelo con mayor fidelidad en este estudio, adaptándose mejor que cualquier otro a los choques estacionales históricos de la economía guatemalteca.
</div>
""", unsafe_allow_html=True)

# --- TABLA PRONÓSTICO SARIMA CENTRADA ---
st.markdown("""
<div style="display:flex; justify-content:center; margin-top:25px;">

<div style="
background-color:#f7f9fc;
padding:25px;
border-radius:10px;
box-shadow:0px 2px 6px rgba(0,0,0,0.08);
width:65%;
">

<h4 style="text-align:center; margin-bottom:0px;">Pronósticos del Flujo de Remesas</h4>
<p style="text-align:center; color:gray; margin-top:3px;">
<i>Modelo SARIMA</i>
</p>

<table style="width:100%; border-collapse: collapse; font-size:15px; text-align:center;">
<thead>
<tr style="background-color:#e9eef6;">
<th style="padding:10px; border-bottom:2px solid #d0d7e2;">Mes-año</th>
<th style="padding:10px; border-bottom:2px solid #d0d7e2;">Flujo de Remesas Familiares (millones USD)</th>
</tr>
</thead>

<tbody>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">abr-24</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,725.19</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">may-24</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,868.50</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">jun-24</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,887.65</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">jul-24</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,840.73</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">ago-24</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,902.59</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">sept-24</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,799.99</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">oct-24</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,879.96</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">nov-24</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,731.84</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">dic-24</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,905.96</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">ene-25</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,619.98</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">feb-25</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,674.07</td></tr>
<tr><td style="padding:8px; border-bottom:1px solid #e1e5eb;">mar-25</td><td style="padding:8px; border-bottom:1px solid #e1e5eb;">1,930.56</td></tr>
</tbody>

</table>
</div>

</div>
""", unsafe_allow_html=True)

# --- RMSE ---
st.markdown("---")
st.subheader("Comparativa de RMSE (Root Mean Square Error)")
rm_cols = st.columns(5)
metrics = [("SARIMA", "110.1685"), ("Originales", "589.4224"), ("P. Móviles", "593.3804"), ("Holt-Winters", "134.9173"), ("Desestac.", "594.86")]
for i, (m, v) in enumerate(metrics):
    rm_cols[i].markdown(f'<div class="metric-container"><b>{m}</b><br><span class="metric-value">{v}</span></div>', unsafe_allow_html=True)

# --- SECCIÓN 2: MODELOS (INTERPRETACIÓN INCLUIDA) ---
st.markdown("---")
st.header("Evaluación de Modelos Simples (2024-2025)")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
if 'sel_mod' not in st.session_state: st.session_state.sel_mod = "HoltWinters"
if col_b1.button("Datos Originales"): st.session_state.sel_mod = "Originales"
if col_b2.button("Promedios Móviles"): st.session_state.sel_mod = "PM"
if col_b3.button("Holt-Winters"): st.session_state.sel_mod = "HoltWinters"
if col_b4.button("Desestacionalización"): st.session_state.sel_mod = "Des"

mod = st.session_state.sel_mod
y24 = df_2024['Divisas'].values
f_dates_24 = pd.date_range(start='2024-04-01', periods=12, freq='MS')
fig2 = go.Figure()
# CAMBIO: Color gris pálido y línea punteada para datos reales
fig2.add_trace(go.Scatter(x=df_2024['Fecha'], y=y24, name="Real", line=dict(color="#D3D3D3", width=1.5, dash='dot')))

# Variables de interpretación vacías
v, d, c = "", "", ""

if mod == "PM":
    pm = np.convolve(y24, np.ones(3)/3, mode='valid')
    pm_c = np.convolve(pm, np.ones(2)/2, mode='valid')
    reg_pm = LinearRegression().fit(np.arange(len(pm_c)).reshape(-1, 1), pm_c)
    f_pm = reg_pm.predict(np.arange(len(pm_c), len(pm_c)+12).reshape(-1, 1))
    fig2.add_trace(go.Scatter(x=df_2024['Fecha'].iloc[2:], y=pm_c, name="Ajuste PM", line=dict(color="#00E676", width=2.5)))
    fig2.add_trace(go.Scatter(x=f_dates_24, y=f_pm, name="Pronóstico", line=dict(color="#00E676", dash='dash')))
    v, d, c = "Suaviza fluctuaciones aleatorias y el ruido de la serie.", "Presenta un rezago respecto a la tendencia actual.", "Este modelo, al regularizar los datos oringinales, realmente no dista demasiado del previo. Las remesas siguen con una tendencia al alza constante. Con un crecimiento mensual de alrededor de cinco millones de dólares."

elif mod == "HoltWinters":
    hw = ExponentialSmoothing(y24, trend="add", seasonal="multiplicative", seasonal_periods=12).fit()
    fig2.add_trace(go.Scatter(x=df_2024['Fecha'], y=hw.fittedvalues, name="Ajuste HW", line=dict(color="#D1C4E9", width=2)))
    fig2.add_trace(go.Scatter(x=f_dates_24, y=hw.forecast(12), name="HW Pronóstico", line=dict(color="#AA00FF", width=3.5)))
    v, d, c = "Modelado integral de tendencia y estacionalidad cambiante.", "Alta sensibilidad a la elección de parámetros iniciales.", "Holt-Winters expresa el comportamiento más lógico, económicamente hablando, que se podría esperar del ingreso de remesas al país. Sigue una tendencia positiva, pero con la estacionalidad representada en su proyección."

elif mod == "Des":
    y_ts = pd.Series(y24, index=pd.date_range(start='2002-01-01', periods=len(y24), freq='M'))
    y_des = y24 / seasonal_decompose(y_ts, model='multiplicative', period=12).seasonal.values
    fig2.add_trace(go.Scatter(x=df_2024['Fecha'], y=y_des, name="Desestacionalizada", line=dict(color="#FF4081", width=2)))
    v, d, c = "Permite identificar el crecimiento subyacente sin ruido mensual.", "No es un modelo predictivo, es una técnica de análisis.", "En este caso, una vez se utiliza el índice estacional, la proyección se torna mucho más coherente con el comportamiento económico de las remesas. Con ciertos meses fuertes y otros débiles."

else: # Datos Originales (Regresión)
    reg = LinearRegression().fit(np.arange(len(y24)).reshape(-1, 1), y24)
    fig2.add_trace(go.Scatter(x=f_dates_24, y=reg.predict(np.arange(len(y24), len(y24)+12).reshape(-1, 1)), name="Lineal", line=dict(color="#FFAB40", width=3)))
    v, d, c = "Extrema simplicidad y fácil interpretación de la pendiente.", "Ignora por completo los ciclos estacionales obligatorios.", "Según este modelo las remesas tendrían una trayectoria de crecimiento constante. Específicamente el modelo proyectado estima un incremento de cinco millones de dólares por mes."

fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

# Cuadros de interpretación para los modelos simples
it1, it2, it3 = st.columns(3)
with it1: st.markdown(f'<div class="interpretation-card"><b>Ventaja:</b><br>{v}</div>', unsafe_allow_html=True)
with it2: st.markdown(f'<div class="interpretation-card"><b>Desventaja:</b><br>{d}</div>', unsafe_allow_html=True)
with it3: st.markdown(f'<div class="interpretation-card"><b>Comportamiento:</b><br>{c}</div>', unsafe_allow_html=True)


# --- SECCIÓN 3: PRONÓSTICO FINAL ---
st.markdown("---")
st.header("Pronóstico Final Holt-Winters (Serie Completa 2002-2027)")
y_f = df_2026['Divisas'].values
model_f = ExponentialSmoothing(y_f, trend="add", seasonal="multiplicative", seasonal_periods=12).fit()
forecast_f = model_f.forecast(12)
rmse_f = np.sqrt(mean_squared_error(y_f, model_f.fittedvalues))
f_idx = pd.date_range(start='2026-02-01', periods=12, freq='MS')

fig3 = go.Figure()
# CAMBIO: Historial en naranja/dorado y pronóstico en verde limón neón
fig3.add_trace(go.Scatter(x=df_2026['Fecha'], y=df_2026['Divisas'], name="Histórico Real", line=dict(color='#ffba44', width=2)))

fig3.add_trace(go.Scatter(x=f_idx, y=forecast_f, name="Pronóstico Feb 26 - Ene 27", line=dict(color="#b5ff3c", width=4)))

# Ajuste de intervalos para que coincidan con el verde #b5ff3c (con 15% de opacidad)
fig3.add_trace(go.Scatter(x=f_idx.tolist() + f_idx.tolist()[::-1], 
                         y=(forecast_f + 1.96*rmse_f).tolist() + (forecast_f - 1.96*rmse_f).tolist()[::-1],
                         fill='toself', 
                         fillcolor='rgba(181, 255, 60, 0.15)', # Este es el verde #b5ff3c en formato RGBA
                         line=dict(color='rgba(255,255,255,0)'), 
                         name="Confianza 95%"))

fig3.update_layout(template="plotly_dark", height=600)
st.plotly_chart(fig3, use_container_width=True)

# TABLA Y JUSTIFICACIÓN
c_tab, c_just = st.columns([1, 1.5])
with c_tab:
    st.markdown("""
    <div class="custom-table-container">
        <div class="table-col">
            <div class="header-blue">Mes / Año</div>
            """ + "".join([f'<div class="cell-blue">{m}</div>' for m in f_idx.strftime('%m - %Y')]) + """
        </div>
        <div class="table-col">
            <div class="header-blue">Pronóstico</div>
            """ + "".join([f'<div class="cell-blue">{v:.6f}</div>' for v in forecast_f]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)

with c_just:
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
