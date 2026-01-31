
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import RESULTS_DIR, PARAMS_SCENARIO_1
from src.io_load import load_m5_data
from src.preprocess import preprocess_data
from src.eda.eda_core import compute_statistics
from src.eda.intermittency import analyze_intermittency
from src.forecast.ets_arima import ETSForecaster
from src.inventory.rq_policy import RQPolicy
from src.inventory.eoq import calculate_eoq

def main():
    print("=== Projeto Logística M5 - Pipeline da Etapa 1 ===")
    
    # 1. Carregar Dados
    sales_raw, calendar, prices = load_m5_data()
    
    # 2. Pré-processar
    print("\n[Passo 2] Pré-processando Dados...")
    df = preprocess_data(sales_raw, calendar, prices)
    
    # Salvar snippet processado para inspeção?
    # df.to_csv(RESULTS_DIR / "processed_snippet.csv", index=False)
    
    # 3. EDA
    print("\n[Passo 3] Rodando EDA...")
    stats = compute_statistics(df)
    print("Top 5 Itens por Volatilidade de Demanda (CV):")
    print(stats.sort_values('cv', ascending=False).head())
    
    intermit = analyze_intermittency(df)
    print("\nResumo da Classificação de Intermitência:")
    print(intermit['classification'].value_counts())
    
    # 4. Previsão e Estoque (Amostra no item principal)
    print("\n[Passo 4 & 5] Piloto de Previsão & Estoque...")
    
    # Escolher um item-loja
    sample_item = df['item_id'].unique()[0]
    sample_store = df['store_id'].unique()[0]
    print(f"Analisando Item: {sample_item} na Loja: {sample_store}")
    
    subset = df[(df['item_id'] == sample_item) & (df['store_id'] == sample_store)].sort_values('date')
    
    # Divisão Treino/Teste
    train = subset.iloc[:-28]
    test = subset.iloc[-28:]
    
    # Previsão (ETS)
    forecaster = ETSForecaster(seasonal_periods=7)
    forecaster.fit(train['demand'])
    pred = forecaster.predict(28)
    
    # Parâmetros de Estoque
    forecast_mean = np.mean(pred)
    forecast_sigma = np.std(subset['demand']) # Proxy simples para desvio padrão da demanda (idealmente resíduo)
    # Idealmente sigma deve ser "RMSE da previsão" no conjunto de validação, ou std do resíduo.
    # Vamos usar std do resíduo do fit de treino se possível, ou apenas std da demanda para EOQ.
    
    # EOQ
    D_annual = subset['demand'].sum() * (365 / len(subset))
    params = PARAMS_SCENARIO_1
    unit_cost = subset['sell_price'].mean()
    h = params['h_pct'] * unit_cost
    
    eoq = calculate_eoq(D_annual, params['K'], h)
    
    # (R,Q)
    policy = RQPolicy(lead_time_days=3, csl_target=0.95)
    rq_params = policy.calculate_parameters(forecast_mean, forecast_sigma)
    
    print(f"\n--- Resultados para {sample_item} ---")
    print(f"Demanda Anual Est: {D_annual:.2f}")
    print(f"EOQ: {eoq:.2f}")
    print(f"Estoque de Segurança: {rq_params['SS']:.2f}")
    print(f"Ponto de Ressuprimento (R): {rq_params['R']:.2f}")
    
    print("\n=== Pipeline Concluído ===")

if __name__ == "__main__":
    main()
