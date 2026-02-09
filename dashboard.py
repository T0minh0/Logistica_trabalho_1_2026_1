"""
Dashboard Interativo - Projeto Logística Quantitativa M5
Execute com: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from src.io_load import load_m5_data
from src.preprocess import preprocess_data
from src.eda.eda_core import compute_statistics
from src.eda.intermittency import analyze_intermittency
from src.forecast.ets_arima import ETSForecaster
from src.forecast.metrics import evaluate_forecast
from src.inventory.eoq import calculate_eoq, total_cost_deterministic
from src.inventory.rq_policy import RQPolicy
from src.config import PARAMS_SCENARIO_1

# Import advanced models
try:
    from src.forecast.advanced import LagFeatureForecaster, EnsembleForecaster, SeasonalDecompForecaster
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False

# ============ CONFIGURAÇÃO DA PÁGINA ============
st.set_page_config(
    page_title="Logística M5 - Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ TEMA GLOBAL (DARK + VERMELHO) ============
pio.templates.default = "plotly_dark"
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#ff4d57", "#c1121f", "#8c1118", "#ff8a92"]
px.defaults.color_continuous_scale = px.colors.sequential.Reds

# ============ ESTILO CSS CUSTOMIZADO ============
st.markdown("""
<style>
    :root {
        --bg-main: #0d0d0f;
        --bg-surface: #17171b;
        --bg-card: #1f1f24;
        --accent: #c1121f;
        --text-main: #f4f4f5;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, rgba(193, 18, 31, 0.18), transparent 35%), var(--bg-main);
        color: var(--text-main);
    }
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #111115 0%, #17171b 100%);
        border-right: 1px solid rgba(193, 18, 31, 0.35);
    }
    [data-testid="stHeader"] {
        background: rgba(13, 13, 15, 0.75);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff4d57 0%, #c1121f 55%, #8c1118 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #2a1115 0%, #511218 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 18px rgba(193, 18, 31, 0.3);
    }
    [data-testid="stMetric"] {
        background-color: var(--bg-card);
        padding: 1rem;
        border-radius: 0.6rem;
        border-left: 4px solid var(--accent);
    }
    .success-metric {
        background-color: rgba(25, 135, 84, 0.15);
        border-left: 4px solid #22c55e;
    }
    .param-info {
        background: rgba(193, 18, 31, 0.14);
        border: 1px solid rgba(193, 18, 31, 0.35);
        padding: 0.6rem;
        border-radius: 0.6rem;
        font-size: 0.82rem;
        margin-top: 0.5rem;
        color: var(--text-main);
    }
    .stAlert {
        background-color: var(--bg-surface);
        color: var(--text-main);
        border: 1px solid rgba(193, 18, 31, 0.35);
    }
    hr {
        border: none;
        border-top: 1px solid rgba(193, 18, 31, 0.28);
    }
</style>
""", unsafe_allow_html=True)

# ============ CACHE DE DADOS ============
@st.cache_data(show_spinner="Carregando dados M5...")
def load_data():
    sales_raw, calendar, prices = load_m5_data()
    df = preprocess_data(sales_raw, calendar, prices)
    return df

@st.cache_data
def get_statistics(df):
    return compute_statistics(df)

@st.cache_data
def get_intermittency(df):
    return analyze_intermittency(df)

# ============ SIDEBAR ============
with st.sidebar:
    st.image("Flamengo.png", width=95)
    st.markdown("## 📊 Navegação")
    
    page = st.radio(
        "Selecione a Página:",
        ["🏠 Visão Geral", "📈 Análise de Demanda", "🔮 Previsão", "📦 Gestão de Estoques", "🎲 Simulação & Etapa 2"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Parâmetros Globais")
    
    # Lead Time - afeta múltiplas páginas
    lead_time = st.slider(
        "Lead Time (dias)", 
        min_value=1, 
        max_value=14, 
        value=3,
        help="Tempo de ressuprimento. Afeta: previsão, estoque de segurança, ponto de ressuprimento."
    )
    
    # Nível de Serviço - afeta estoque de segurança
    csl_target = st.slider(
        "Nível de Serviço (CSL)", 
        min_value=0.80, 
        max_value=0.99, 
        value=0.95, 
        step=0.01,
        help="Probabilidade de não faltar estoque. Valores maiores = mais estoque de segurança."
    )
    
    # Horizonte de previsão
    default_horizon = st.selectbox(
        "Horizonte de Previsão Padrão",
        options=[7, 14, 21, 28],
        index=2,
        help="Número de dias para projetar a demanda futura."
    )
    
    st.markdown("---")
    st.markdown("### 💰 Parâmetros Econômicos")
    
    # Custo de Pedido
    K_cost = st.number_input(
        "Custo por Pedido (K)", 
        min_value=10.0, 
        max_value=500.0, 
        value=float(PARAMS_SCENARIO_1['K']),
        step=10.0,
        help="Custo fixo de setup por pedido (R$)."
    )
    
    # Holding Cost %
    h_pct = st.slider(
        "Custo de Holding (% valor)", 
        min_value=0.10, 
        max_value=0.50, 
        value=float(PARAMS_SCENARIO_1['h_pct']),
        step=0.05,
        help="Custo anual de manter estoque como % do valor do item."
    )
    
    st.markdown("---")
    st.markdown("### 📁 Dataset")
    st.info("M5 Forecasting (Walmart)")
    
    # Mostrar resumo dos parâmetros ativos
    st.markdown("---")
    st.markdown("### 📋 Parâmetros Ativos")
    st.markdown(f"""
    <div class="param-info">
    <b>Lead Time:</b> {lead_time} dias<br>
    <b>CSL:</b> {csl_target:.0%}<br>
    <b>Horizonte:</b> {default_horizon} dias<br>
    <b>K:</b> R$ {K_cost:.0f}<br>
    <b>h:</b> {h_pct:.0%} a.a.
    </div>
    """, unsafe_allow_html=True)

# ============ CARREGAMENTO DE DADOS ============
try:
    df = load_data()
    stats = get_statistics(df)
    intermit = get_intermittency(df)
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# ============ PÁGINAS ============

if page == "🏠 Visão Geral":
    st.markdown('<h1 class="main-header">📦 Logística Quantitativa M5</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Info sobre parâmetros
    st.info(f"📊 **Parâmetros ativos:** Lead Time = {lead_time} dias | CSL = {csl_target:.0%} | Horizonte = {default_horizon} dias | K = R${K_cost:.0f} | h = {h_pct:.0%}")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏪 Lojas", df['store_id'].nunique())
    with col2:
        st.metric("📦 Itens", df['item_id'].nunique())
    with col3:
        st.metric("📅 Dias", df['date'].nunique())
    with col4:
        st.metric("📊 Registros", f"{len(df):,}")
    
    st.markdown("---")
    
    # Métricas de estoque agregadas (usando parâmetros)
    st.markdown("### 📦 Visão Agregada de Estoque (baseada nos parâmetros)")
    
    # Calcular métricas agregadas
    total_demand = df['demand'].sum()
    avg_daily_demand = df.groupby('date')['demand'].sum().mean()
    demand_std = df.groupby('date')['demand'].sum().std()
    
    # Calcular estoque de segurança agregado
    from scipy.stats import norm
    z_score = norm.ppf(csl_target)
    sigma_L = demand_std * np.sqrt(lead_time)
    ss_total = z_score * sigma_L
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Demanda Média/Dia", f"{avg_daily_demand:,.0f} un")
    with col2:
        st.metric(f"Demanda no Lead Time ({lead_time}d)", f"{avg_daily_demand * lead_time:,.0f} un")
    with col3:
        st.metric(f"SS Agregado (CSL={csl_target:.0%})", f"{ss_total:,.0f} un")
    with col4:
        st.metric("z-score", f"{z_score:.2f}")
    
    st.markdown("---")
    
    # Gráficos de visão geral
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Demanda Total por Loja")
        demand_by_store = df.groupby('store_id')['demand'].sum().reset_index()
        fig = px.bar(
            demand_by_store, 
            x='store_id', 
            y='demand',
            color='demand',
            color_continuous_scale='Viridis',
            labels={'demand': 'Demanda Total', 'store_id': 'Loja'}
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### 🎯 Classificação de Intermitência")
        intermit_counts = intermit['classification'].value_counts().reset_index()
        intermit_counts.columns = ['Classificação', 'Contagem']
        fig = px.pie(
            intermit_counts, 
            values='Contagem', 
            names='Classificação',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Série temporal agregada
    st.markdown("### 📊 Demanda Agregada ao Longo do Tempo")
    daily_demand = df.groupby('date')['demand'].sum().reset_index()
    fig = px.area(
        daily_demand, 
        x='date', 
        y='demand',
        labels={'demand': 'Demanda Total', 'date': 'Data'},
        color_discrete_sequence=['#ff4d57']
    )
    # Adicionar linha do ponto de ressuprimento agregado
    fig.add_hline(y=avg_daily_demand * lead_time + ss_total, 
                  line_dash="dash", 
                  line_color="red",
                  annotation_text=f"Nível de Ressuprimento (R={avg_daily_demand * lead_time + ss_total:,.0f})")
    fig.update_layout(height=350)
    st.plotly_chart(fig, width='stretch')

elif page == "📈 Análise de Demanda":
    st.markdown('<h1 class="main-header">📈 Análise de Demanda</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Info sobre parâmetros
    st.info(f"📊 **Horizonte de análise:** {default_horizon} dias | **Lead Time:** {lead_time} dias")
    
    # Seletores
    col1, col2 = st.columns(2)
    with col1:
        selected_store = st.selectbox("🏪 Selecione a Loja", df['store_id'].unique())
    with col2:
        items_in_store = df[df['store_id'] == selected_store]['item_id'].unique()
        selected_item = st.selectbox("📦 Selecione o Item", items_in_store)
    
    # Filtrar dados
    subset = df[(df['store_id'] == selected_store) & (df['item_id'] == selected_item)].sort_values('date')
    
    if len(subset) > 0:
        # Série temporal do item
        st.markdown(f"### 📊 Série Temporal: {selected_item}")
        fig = px.line(
            subset, 
            x='date', 
            y='demand',
            labels={'demand': 'Demanda Diária', 'date': 'Data'},
            color_discrete_sequence=['#c1121f']
        )
        fig.add_scatter(
            x=subset['date'], 
            y=subset['demand'].rolling(7).mean(),
            name='Média Móvel (7d)',
            line=dict(color='#ff4d57', width=2)
        )
        # Adicionar linha de demanda média durante lead time
        mean_demand = subset['demand'].mean()
        fig.add_hline(y=mean_demand, line_dash="dot", line_color="green",
                      annotation_text=f"Média = {mean_demand:.1f}")
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width='stretch')
        
        # Estatísticas (usando parâmetros)
        from scipy.stats import norm
        z_score = norm.ppf(csl_target)
        mean_d = subset['demand'].mean()
        std_d = subset['demand'].std()
        sigma_L = std_d * np.sqrt(lead_time)
        ss = z_score * sigma_L
        reorder_point = mean_d * lead_time + ss
        
        st.markdown("### 📊 Estatísticas & Parâmetros de Estoque")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Média Diária (μ)", f"{mean_d:.2f}")
            st.metric("Desvio Padrão (σ)", f"{std_d:.2f}")
        with col2:
            cv = std_d / mean_d if mean_d > 0 else 0
            zero_pct = (subset['demand'] == 0).mean()
            st.metric("CV", f"{cv:.2%}")
            st.metric("% Zeros", f"{zero_pct:.1%}")
        with col3:
            st.metric(f"μ durante LT ({lead_time}d)", f"{mean_d * lead_time:.1f}")
            st.metric(f"σ durante LT", f"{sigma_L:.2f}")
        with col4:
            st.metric(f"SS (CSL={csl_target:.0%})", f"{ss:.1f}")
            st.metric("Ponto Ressuprimento (R)", f"{reorder_point:.1f}")
        
        st.markdown("---")
        
        # Heatmap de sazonalidade
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📅 Sazonalidade Semanal")
            weekly = subset.groupby('day_of_week')['demand'].mean().reset_index()
            weekly['dia'] = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
            fig = px.bar(
                weekly, 
                x='dia', 
                y='demand',
                color='demand',
                color_continuous_scale='RdYlGn',
                labels={'demand': 'Demanda Média'}
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("### 📆 Sazonalidade Mensal")
            monthly = subset.groupby('month')['demand'].mean().reset_index()
            fig = px.bar(
                monthly, 
                x='month', 
                y='demand',
                color='demand',
                color_continuous_scale='Plasma',
                labels={'demand': 'Demanda Média', 'month': 'Mês'}
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, width='stretch')

elif page == "🔮 Previsão":
    st.markdown('<h1 class="main-header">🔮 Previsão de Demanda</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Info sobre parâmetros
    st.info(f"📊 **Horizonte padrão:** {default_horizon} dias | **Lead Time:** {lead_time} dias (usado para previsão durante ressuprimento)")
    
    # Seletores
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_store = st.selectbox("🏪 Loja", df['store_id'].unique())
    with col2:
        items_in_store = df[df['store_id'] == selected_store]['item_id'].unique()
        selected_item = st.selectbox("📦 Item", items_in_store)
    with col3:
        # Usa horizonte padrão da sidebar, mas permite override
        horizon = st.selectbox("📅 Horizonte (dias)", [7, 14, 21, 28], 
                               index=[7, 14, 21, 28].index(default_horizon) if default_horizon in [7, 14, 21, 28] else 2)
    with col4:
        model_type = st.selectbox("🤖 Modelo", [
            "ETS (Suavização Exponencial)",
            "LightGBM (Machine Learning)",
            "Ensemble (Combinado)",
            "Decomposição Sazonal"
        ])
    
    subset = df[(df['store_id'] == selected_store) & (df['item_id'] == selected_item)].sort_values('date')
    
    if len(subset) > horizon + 60:
        train = subset.iloc[:-horizon]
        test = subset.iloc[-horizon:]
        
        # Treinar modelo selecionado
        with st.spinner(f"Treinando modelo {model_type}..."):
            if model_type == "ETS (Suavização Exponencial)":
                forecaster = ETSForecaster(seasonal_periods=7)
                forecaster.fit(train['demand'])
                predictions = forecaster.predict(horizon)
                
            elif model_type == "LightGBM (Machine Learning)" and HAS_ADVANCED:
                forecaster = LagFeatureForecaster()
                forecaster.fit(train['demand'])
                predictions = forecaster.predict(horizon)
                
            elif model_type == "Ensemble (Combinado)" and HAS_ADVANCED:
                forecaster = EnsembleForecaster()
                forecaster.fit(train['demand'])
                predictions = forecaster.predict(horizon)
                
            elif model_type == "Decomposição Sazonal" and HAS_ADVANCED:
                forecaster = SeasonalDecompForecaster(period=7)
                forecaster.fit(train['demand'])
                predictions = forecaster.predict(horizon)
            else:
                # Fallback para ETS
                forecaster = ETSForecaster(seasonal_periods=7)
                forecaster.fit(train['demand'])
                predictions = forecaster.predict(horizon)
        
        # Métricas
        metrics = evaluate_forecast(test['demand'].values, predictions)
        
        # Calcular acurácia (100% - WAPE)
        accuracy = max(0, 1 - metrics['WAPE']) * 100
        
        # Métricas de previsão durante Lead Time
        lt_predictions = predictions[:lead_time] if len(predictions) >= lead_time else predictions
        lt_mean = np.mean(lt_predictions)
        lt_sum = np.sum(lt_predictions)
        
        st.markdown("### 📊 Métricas de Desempenho")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
        with col2:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        with col3:
            st.metric("WAPE", f"{metrics['WAPE']:.2%}")
        with col4:
            st.metric("SMAPE", f"{metrics['SMAPE']:.1f}%")
        with col5:
            delta_color = "normal" if accuracy >= 80 else "inverse"
            st.metric("Acurácia", f"{accuracy:.1f}%", delta=f"{'Bom' if accuracy >= 80 else 'Baixo'}", delta_color=delta_color)
        
        # Métricas para estoque (usando lead time)
        st.markdown(f"### 📦 Previsão para Estoque (Lead Time = {lead_time} dias)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Demanda Prevista no LT", f"{lt_sum:.1f} un")
        with col2:
            st.metric(f"Média Diária Prevista", f"{lt_mean:.2f} un")
        with col3:
            # Calcular SS baseado em resíduos
            residuals = test['demand'].values - predictions
            sigma_resid = np.std(residuals)
            from scipy.stats import norm
            z = norm.ppf(csl_target)
            ss_forecast = z * sigma_resid * np.sqrt(lead_time)
            st.metric(f"SS Recomendado (CSL={csl_target:.0%})", f"{ss_forecast:.1f} un")
        
        st.markdown("---")
        
        # Gráfico de previsão
        st.markdown("### 📊 Previsão vs Real")
        
        fig = go.Figure()
        
        # Histórico
        fig.add_trace(go.Scatter(
            x=train['date'].tail(60),
            y=train['demand'].tail(60),
            name='Histórico',
            line=dict(color='#ff4d57', width=2)
        ))
        
        # Real
        fig.add_trace(go.Scatter(
            x=test['date'],
            y=test['demand'],
            name='Real',
            line=dict(color='#f5f5f5', width=2)
        ))
        
        # Previsão
        fig.add_trace(go.Scatter(
            x=test['date'],
            y=predictions,
            name=f'Previsão ({model_type.split(" ")[0]})',
            line=dict(color='#c1121f', width=2, dash='dash')
        ))
        
        # Destacar período de Lead Time
        if len(test) >= lead_time:
            fig.add_vrect(
                x0=test['date'].iloc[0],
                x1=test['date'].iloc[lead_time-1],
                fillcolor="rgba(193, 18, 31, 0.16)",
                layer="below",
                line_width=0,
                annotation_text=f"Lead Time ({lead_time}d)",
                annotation_position="top left"
            )
        
        # Intervalo de confiança (aproximado)
        std_pred = np.std(train['demand'].tail(28))
        fig.add_trace(go.Scatter(
            x=list(test['date']) + list(test['date'][::-1]),
            y=list(predictions + 1.96*std_pred) + list((predictions - 1.96*std_pred)[::-1]),
            fill='toself',
            fillcolor='rgba(193, 18, 31, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            name='IC 95%'
        ))
        
        fig.update_layout(
            height=450,
            legend=dict(orientation="h", y=-0.15),
            xaxis_title="Data",
            yaxis_title="Demanda"
        )
        st.plotly_chart(fig, width='stretch')
        
        # Comparação de modelos
        if HAS_ADVANCED:
            st.markdown("---")
            st.markdown("### 🏆 Comparação de Modelos")
            
            with st.spinner("Comparando todos os modelos..."):
                model_results = []
                
                # ETS
                try:
                    ets = ETSForecaster(seasonal_periods=7)
                    ets.fit(train['demand'])
                    ets_pred = ets.predict(horizon)
                    ets_metrics = evaluate_forecast(test['demand'].values, ets_pred)
                    model_results.append({
                        'Modelo': 'ETS',
                        'MAE': ets_metrics['MAE'],
                        'RMSE': ets_metrics['RMSE'],
                        'WAPE': ets_metrics['WAPE'],
                        'Acurácia': max(0, (1 - ets_metrics['WAPE'])) * 100
                    })
                except:
                    pass
                
                # LightGBM
                try:
                    lgbm = LagFeatureForecaster()
                    lgbm.fit(train['demand'])
                    lgbm_pred = lgbm.predict(horizon)
                    lgbm_metrics = evaluate_forecast(test['demand'].values, lgbm_pred)
                    model_results.append({
                        'Modelo': 'LightGBM',
                        'MAE': lgbm_metrics['MAE'],
                        'RMSE': lgbm_metrics['RMSE'],
                        'WAPE': lgbm_metrics['WAPE'],
                        'Acurácia': max(0, (1 - lgbm_metrics['WAPE'])) * 100
                    })
                except:
                    pass
                
                # Ensemble
                try:
                    ens = EnsembleForecaster()
                    ens.fit(train['demand'])
                    ens_pred = ens.predict(horizon)
                    ens_metrics = evaluate_forecast(test['demand'].values, ens_pred)
                    model_results.append({
                        'Modelo': 'Ensemble',
                        'MAE': ens_metrics['MAE'],
                        'RMSE': ens_metrics['RMSE'],
                        'WAPE': ens_metrics['WAPE'],
                        'Acurácia': max(0, (1 - ens_metrics['WAPE'])) * 100
                    })
                except:
                    pass
                
                # Decomp
                try:
                    decomp = SeasonalDecompForecaster(period=7)
                    decomp.fit(train['demand'])
                    decomp_pred = decomp.predict(horizon)
                    decomp_metrics = evaluate_forecast(test['demand'].values, decomp_pred)
                    model_results.append({
                        'Modelo': 'Decomposição',
                        'MAE': decomp_metrics['MAE'],
                        'RMSE': decomp_metrics['RMSE'],
                        'WAPE': decomp_metrics['WAPE'],
                        'Acurácia': max(0, (1 - decomp_metrics['WAPE'])) * 100
                    })
                except:
                    pass
                
                if model_results:
                    results_df = pd.DataFrame(model_results)
                    results_df = results_df.sort_values('Acurácia', ascending=False)
                    
                    # Gráfico de barras
                    fig = px.bar(
                        results_df,
                        x='Modelo',
                        y='Acurácia',
                        color='Acurácia',
                        color_continuous_scale='RdYlGn',
                        text='Acurácia'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Tabela detalhada
                    with st.expander("📋 Ver Métricas Detalhadas"):
                        st.dataframe(results_df.style.format({
                            'MAE': '{:.2f}',
                            'RMSE': '{:.2f}',
                            'WAPE': '{:.2%}',
                            'Acurácia': '{:.1f}%'
                        }), width='stretch')

elif page == "📦 Gestão de Estoques":
    st.markdown('<h1 class="main-header">📦 Gestão de Estoques</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Info sobre parâmetros
    st.info(f"📊 **Parâmetros:** Lead Time = {lead_time} dias | CSL = {csl_target:.0%} | K = R${K_cost:.0f} | h = {h_pct:.0%} a.a.")
    
    # Seletores
    col1, col2 = st.columns(2)
    with col1:
        selected_store = st.selectbox("🏪 Loja", df['store_id'].unique(), key="inv_store")
    with col2:
        items_in_store = df[df['store_id'] == selected_store]['item_id'].unique()
        selected_item = st.selectbox("📦 Item", items_in_store, key="inv_item")
    
    subset = df[(df['store_id'] == selected_store) & (df['item_id'] == selected_item)].sort_values('date')
    
    if len(subset) > 0:
        # Parâmetros (usando valores da sidebar)
        D_annual = subset['demand'].sum() * (365 / len(subset))
        unit_cost = subset['sell_price'].mean()
        h = h_pct * unit_cost  # Usa parâmetro da sidebar
        K = K_cost  # Usa parâmetro da sidebar
        
        # EOQ
        eoq = calculate_eoq(D_annual, K, h)
        total_cost = total_cost_deterministic(eoq, D_annual, K, h)
        
        # Política (R, Q) usando parâmetros da sidebar
        policy = RQPolicy(lead_time_days=lead_time, csl_target=csl_target)
        forecast_mean = subset['demand'].mean()
        forecast_sigma = subset['demand'].std()
        rq_params = policy.calculate_parameters(forecast_mean, forecast_sigma)
        
        st.markdown("### 📊 Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 EOQ (Determinístico)")
            st.metric("Demanda Anual", f"{D_annual:,.0f} un")
            st.metric("Lote Econômico (Q*)", f"{eoq:,.0f} un")
            st.metric("Custo Total Anual", f"R$ {total_cost:,.2f}")
            st.metric("Número de Pedidos/Ano", f"{D_annual/eoq:,.1f}")
            st.caption(f"*K = R${K:.0f}, h = R${h:.2f}/un/ano*")
        
        with col2:
            st.markdown("#### 📈 Política (R, Q) Estocástica")
            st.metric("Estoque de Segurança", f"{rq_params['SS']:,.1f} un")
            st.metric("Ponto de Ressuprimento (R)", f"{rq_params['R']:,.1f} un")
            st.metric("Demanda Média no LT", f"{rq_params['mu_L']:,.1f} un")
            st.metric("σ Demanda no LT", f"{rq_params['sigma_L']:,.1f} un")
            st.caption(f"*Lead Time = {lead_time} dias, CSL = {csl_target:.0%}*")
        
        st.markdown("---")
        
        # Análise de Sensibilidade por CSL
        st.markdown("### 📈 Análise de Sensibilidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sensibilidade ao CSL")
            from scipy.stats import norm
            csl_range = np.arange(0.80, 0.995, 0.01)
            ss_values = [norm.ppf(c) * rq_params['sigma_L'] for c in csl_range]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=csl_range * 100,
                y=ss_values,
                name='Estoque de Segurança',
                line=dict(color='#ff4d57', width=3)
            ))
            fig.add_vline(x=csl_target * 100, line_dash="dash", line_color="#c1121f",
                          annotation_text=f"CSL atual = {csl_target:.0%}")
            fig.update_layout(
                height=350,
                xaxis_title="Nível de Serviço (%)",
                yaxis_title="Estoque de Segurança (un)"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### Sensibilidade ao Lead Time")
            lt_range = np.arange(1, 15)
            ss_lt_values = [norm.ppf(csl_target) * forecast_sigma * np.sqrt(lt) for lt in lt_range]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=lt_range,
                y=ss_lt_values,
                name='Estoque de Segurança',
                line=dict(color='#c1121f', width=3)
            ))
            fig.add_vline(x=lead_time, line_dash="dash", line_color="#c1121f",
                          annotation_text=f"LT atual = {lead_time}d")
            fig.update_layout(
                height=350,
                xaxis_title="Lead Time (dias)",
                yaxis_title="Estoque de Segurança (un)"
            )
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # Gráfico de Análise de Sensibilidade EOQ
        st.markdown("### 📉 Curva de Custo Total - EOQ")
        
        Q_range = np.linspace(max(1, eoq*0.3), eoq*2, 100)
        costs = [total_cost_deterministic(q, D_annual, K, h) for q in Q_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=Q_range, 
            y=costs,
            name='Custo Total',
            line=dict(color='#ff4d57', width=3)
        ))
        fig.add_vline(x=eoq, line_dash="dash", line_color="#c1121f", 
                      annotation_text=f"EOQ = {eoq:.0f}")
        fig.update_layout(
            height=400,
            xaxis_title="Quantidade do Pedido (Q)",
            yaxis_title="Custo Total Anual (R$)"
        )
        st.plotly_chart(fig, width='stretch')

elif page == "🎲 Simulação & Etapa 2":
    st.markdown('<h1 class="main-header">🎲 Simulação & Otimização (Etapa 2)</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["🎰 Simulação Monte Carlo", "🔗 Risk Pooling Avançado", "📊 Otimização Multi-Item"])
    
    # ===================== TAB 1: SIMULAÇÃO MONTE CARLO =====================
    with tab1:
        st.markdown("### 🎰 Simulação de Políticas de Estoque com SimPy")
        st.info(f"**Lead Time Estocástico:** μ = {lead_time} dias, σ = {lead_time * 0.3:.1f} dias (Normal Truncada)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_store_sim = st.selectbox("🏪 Loja", df['store_id'].unique(), key="sim_store")
        with col2:
            items_in_store_sim = df[df['store_id'] == selected_store_sim]['item_id'].unique()
            selected_item_sim = st.selectbox("📦 Item", items_in_store_sim, key="sim_item")
        with col3:
            n_replications = st.selectbox("🔄 Replicações Monte Carlo", [10, 30, 50, 100], index=1)
        
        # Parâmetros de simulação
        col1, col2, col3 = st.columns(3)
        with col1:
            sim_horizon = st.number_input("Horizonte (dias)", min_value=90, max_value=730, value=365)
        with col2:
            stockout_cost = st.number_input("Custo de Falta (R$/un)", min_value=1.0, max_value=500.0, value=50.0)
        with col3:
            lt_std_factor = st.slider("Variabilidade LT (%)", 0, 50, 30) / 100
        
        subset_sim = df[(df['store_id'] == selected_store_sim) & (df['item_id'] == selected_item_sim)]
        
        if len(subset_sim) > 0 and st.button("▶️ Executar Simulação Monte Carlo", type="primary"):
            with st.spinner("Executando simulação..."):
                try:
                    from src.simulation.simpy_env import (
                        SimulationConfig, run_monte_carlo, compare_policies,
                        RQPolicy as SimRQPolicy, sSPolicy, PSPolicy,
                        create_policies_from_params
                    )
                    
                    # Configurar simulação
                    demand_mean = subset_sim['demand'].mean()
                    demand_std = subset_sim['demand'].std()
                    unit_cost = subset_sim['sell_price'].mean()
                    
                    config = SimulationConfig(
                        horizon_days=sim_horizon,
                        n_replications=n_replications,
                        demand_mean=demand_mean,
                        demand_std=demand_std,
                        lead_time_mean=float(lead_time),
                        lead_time_std=float(lead_time * lt_std_factor),
                        lead_time_min=max(1, lead_time - 2),
                        lead_time_max=lead_time + 5,
                        ordering_cost=K_cost,
                        holding_cost_rate=h_pct,
                        unit_cost=unit_cost,
                        stockout_cost=stockout_cost,
                        csl_target=csl_target
                    )
                    
                    # Criar políticas
                    policies = create_policies_from_params(
                        demand_mean, demand_std,
                        lead_time, lead_time * lt_std_factor,
                        csl_target, K_cost, h_pct * unit_cost
                    )
                    
                    # Executar comparação
                    results_df = compare_policies(config, policies[:3])  # (R,Q), (s,S), (P,S)
                    
                    st.success(f"✅ Simulação concluída: {n_replications} replicações × {len(policies[:3])} políticas")
                    
                    # Resultados
                    st.markdown("### 📊 Comparação de Políticas")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gráfico de custo
                        fig = px.bar(
                            results_df,
                            x='Política',
                            y='Custo Total (média)',
                            color='Custo Total (média)',
                            color_continuous_scale='RdYlGn_r',
                            text='Custo Total (média)'
                        )
                        fig.update_traces(texttemplate='R$ %{text:,.0f}', textposition='outside')
                        fig.update_layout(height=350, showlegend=False, yaxis_title="Custo Total (R$)")
                        st.plotly_chart(fig, width='stretch')
                    
                    with col2:
                        # Gráfico de fill rate
                        fig = px.bar(
                            results_df,
                            x='Política',
                            y='Fill Rate (%)',
                            color='Fill Rate (%)',
                            color_continuous_scale='RdYlGn',
                            text='Fill Rate (%)'
                        )
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig.update_layout(height=350, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                    
                    # Tabela detalhada
                    st.markdown("### 📋 Resultados Detalhados")
                    st.dataframe(results_df.style.format({
                        'Custo Total (média)': 'R$ {:,.2f}',
                        'Fill Rate (%)': '{:.1f}%',
                        'Estoque Médio': '{:,.1f}',
                        'Faltas Totais': '{:,.0f}',
                        'Custo Holding': 'R$ {:,.2f}',
                        'Custo Pedidos': 'R$ {:,.2f}',
                        'Custo Falta': 'R$ {:,.2f}'
                    }), width='stretch')
                    
                    # Recomendação
                    best_policy = results_df.loc[results_df['Custo Total (média)'].idxmin(), 'Política']
                    st.success(f"🏆 **Política Recomendada:** {best_policy}")
                    
                except Exception as e:
                    st.error(f"Erro na simulação: {e}")
                    st.exception(e)
    
    # ===================== TAB 2: RISK POOLING AVANÇADO =====================
    with tab2:
        st.markdown("### 🔗 Análise de Risk Pooling entre Lojas")
        
        store_list = df['store_id'].unique().tolist()
        selected_stores = st.multiselect(
            "Selecione lojas para análise de pooling:",
            store_list,
            default=store_list[:min(3, len(store_list))]
        )
        
        if len(selected_stores) >= 2:
            if st.button("📊 Calcular Risk Pooling", type="primary"):
                with st.spinner("Calculando correlação e redução de SS..."):
                    try:
                        from src.inventory.pooling_advanced import (
                            calculate_correlation_matrix,
                            analyze_correlation_impact,
                            calculate_ss_reduction,
                            abc_classification,
                            hybrid_pooling_scenario,
                            pooling_sensitivity_analysis
                        )
                        
                        # Matriz de correlação
                        pivot, corr_matrix = calculate_correlation_matrix(df, selected_stores)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📈 Matriz de Correlação entre Lojas")
                            fig = px.imshow(
                                corr_matrix,
                                x=corr_matrix.columns,
                                y=corr_matrix.index,
                                color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1,
                                text_auto='.2f'
                            )
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, width='stretch')
                        
                        with col2:
                            corr_stats = analyze_correlation_impact(corr_matrix)
                            st.markdown("#### 📊 Estatísticas de Correlação")
                            st.metric("Correlação Média", f"{corr_stats['mean_correlation']:.2f}")
                            st.metric("Potencial de Pooling", f"{corr_stats['pooling_potential']:.0%}")
                            st.metric("Pares Altamente Correlacionados", corr_stats['highly_correlated_pairs'])
                        
                        st.markdown("---")
                        
                        # Redução de SS
                        ss_analysis = calculate_ss_reduction(
                            df, selected_stores, lead_time, csl_target
                        )
                        
                        st.markdown("### 📦 Redução de Estoque de Segurança")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("SS Descentralizado", f"{ss_analysis['ss_decentralized']:,.0f} un")
                        with col2:
                            st.metric("SS Centralizado", f"{ss_analysis['ss_centralized']:,.0f} un")
                        with col3:
                            st.metric("Redução", f"{ss_analysis['ss_reduction']:,.0f} un", 
                                     delta=f"-{ss_analysis['ss_reduction_pct']:.1f}%")
                        with col4:
                            st.metric("Portfolio Effect Teórico", f"{ss_analysis['portfolio_effect_theoretical']:.1f}%")
                        
                        st.markdown("---")
                        
                        # Cenário Híbrido ABC
                        st.markdown("### 🔄 Cenário Híbrido ABC")
                        
                        abc_df = abc_classification(df)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Gráfico ABC
                            abc_summary = abc_df.groupby('class').agg({
                                'item_id': 'count',
                                'value_pct': 'sum'
                            }).reset_index()
                            abc_summary.columns = ['Classe', 'Itens', '% Valor']
                            
                            fig = px.bar(
                                abc_summary,
                                x='Classe',
                                y='% Valor',
                                color='Classe',
                                color_discrete_map={'A': '#ff8a92', 'B': '#ff4d57', 'C': '#8c1118'},
                                text='% Valor'
                            )
                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, width='stretch')
                        
                        with col2:
                            st.markdown("**Classificação ABC:**")
                            for _, row in abc_summary.iterrows():
                                st.write(f"- **Classe {row['Classe']}:** {row['Itens']} itens ({row['% Valor']:.1f}% do valor)")
                            
                            st.markdown("""
                            **Estratégia Híbrida:**
                            - 🟢 **A**: Descentralizado (alto giro, rápido)
                            - 🟡 **B/C**: Centralizado (pooling reduz SS)
                            """)
                        
                        # Sensibilidade
                        st.markdown("---")
                        st.markdown("### 📈 Sensibilidade por CSL")
                        
                        sens_df = pooling_sensitivity_analysis(df, selected_stores, lead_time)
                        st.dataframe(sens_df.style.format({
                            'SS Descentralizado': '{:,.0f}',
                            'SS Centralizado': '{:,.0f}',
                            'Redução (un)': '{:,.0f}',
                            'Redução (%)': '{:.1f}%',
                            'Correlação Média': '{:.2f}'
                        }), width='stretch')
                        
                    except Exception as e:
                        st.error(f"Erro na análise: {e}")
                        st.exception(e)
        else:
            st.warning("Selecione pelo menos 2 lojas para análise de pooling.")
    
    # ===================== TAB 3: OTIMIZAÇÃO MULTI-ITEM =====================
    with tab3:
        st.markdown("### 📊 Otimização com Custo de Falta e Restrição de Orçamento")
        
        col1, col2 = st.columns(2)
        with col1:
            stockout_cost_opt = st.number_input("Custo de Falta (p) R$/un", min_value=1.0, max_value=1000.0, value=100.0)
            budget = st.number_input("Orçamento para Estoque (R$)", min_value=1000.0, max_value=1000000.0, value=50000.0)
        with col2:
            selected_store_opt = st.selectbox("🏪 Loja para Otimização", df['store_id'].unique(), key="opt_store")
            n_items_opt = st.slider("Número de itens top", 5, 50, 20)
        
        if st.button("🎯 Otimizar Q* com Custo de Falta", type="primary"):
            with st.spinner("Calculando..."):
                try:
                    from src.inventory.optimization import (
                        optimal_Q_with_stockout,
                        sensitivity_analysis_stockout_cost,
                        multi_item_budget_constraint,
                        ItemParams
                    )
                    
                    # Pegar top itens
                    store_df = df[df['store_id'] == selected_store_opt]
                    top_items = store_df.groupby('item_id')['demand'].sum().nlargest(n_items_opt).index.tolist()
                    
                    # Exemplo com 1 item
                    sample_item = top_items[0]
                    sample_df = store_df[store_df['item_id'] == sample_item]
                    
                    D = sample_df['demand'].mean() * 365
                    sigma = sample_df['demand'].std()
                    unit_cost = sample_df['sell_price'].mean()
                    h = h_pct * unit_cost
                    
                    result = optimal_Q_with_stockout(D, K_cost, h, stockout_cost_opt, sigma, lead_time)
                    
                    st.markdown(f"### 🎯 Resultado para {sample_item}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Q* Ótimo", f"{result['Q_star']:,.0f} un")
                        st.metric("Ponto de Ressuprimento (R)", f"{result['R']:,.0f} un")
                    with col2:
                        st.metric("CSL Ótimo", f"{result['csl_optimal']:.1%}")
                        st.metric("Estoque de Segurança", f"{result['SS']:,.0f} un")
                    with col3:
                        st.metric("Custo Total", f"R$ {result['total_cost']:,.0f}")
                        st.metric("Fill Rate Esperado", f"{result['fill_rate']:.1%}")
                    
                    st.markdown("---")
                    
                    # Sensibilidade ao custo de falta
                    st.markdown("### 📈 Sensibilidade ao Custo de Falta")
                    
                    sens_df = sensitivity_analysis_stockout_cost(D, K_cost, h, sigma, lead_time)
                    
                    fig = make_subplots(rows=1, cols=2, subplot_titles=('Q* vs Custo Falta', 'CSL Ótimo vs Custo Falta'))
                    
                    fig.add_trace(
                        go.Scatter(x=sens_df['p/h Ratio'], y=sens_df['Q*'], mode='lines+markers', name='Q*'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=sens_df['p/h Ratio'], y=sens_df['CSL Ótimo (%)'], mode='lines+markers', name='CSL'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, width='stretch')
                    
                    st.markdown("---")
                    
                    # Multi-item com restrição de orçamento
                    st.markdown("### 💰 Alocação Multi-Item com Restrição de Orçamento")
                    
                    items_list = []
                    for item_id in top_items[:10]:  # Limitar a 10 para performance
                        item_df = store_df[store_df['item_id'] == item_id]
                        items_list.append(ItemParams(
                            item_id=item_id,
                            demand_mean=item_df['demand'].mean(),
                            demand_std=item_df['demand'].std(),
                            unit_cost=item_df['sell_price'].mean(),
                            lead_time=lead_time
                        ))
                    
                    allocation = multi_item_budget_constraint(items_list, K_cost, h_pct, stockout_cost_opt, budget, csl_target)
                    
                    st.info(f"**Status:** {allocation['status']} | **Utilização do Orçamento:** {allocation['budget_utilization']:.1f}%")
                    
                    alloc_df = allocation['allocation'][['item_id', 'Q_eoq', 'SS', 'inventory_value', 'allocation_pct']]
                    alloc_df.columns = ['Item', 'Q (EOQ)', 'SS', 'Valor Estoque (R$)', 'Alocação (%)']
                    
                    st.dataframe(alloc_df.style.format({
                        'Q (EOQ)': '{:,.0f}',
                        'SS': '{:,.1f}',
                        'Valor Estoque (R$)': 'R$ {:,.2f}',
                        'Alocação (%)': '{:.1f}%'
                    }), width='stretch')
                    
                except Exception as e:
                    st.error(f"Erro na otimização: {e}")
                    st.exception(e)

# ============ FOOTER ============
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 1rem;'>
        📦 Projeto Logística Quantitativa - M5 Forecasting | UnB
    </div>
    """, 
    unsafe_allow_html=True
)
