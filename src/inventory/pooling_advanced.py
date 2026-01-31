"""
Análise Avançada de Risk Pooling
Mede correlação entre lojas, calcula redução de SS e cenários híbridos ABC.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PoolingScenario:
    """Resultado de um cenário de pooling."""
    name: str
    ss_total: float
    ss_reduction_pct: float
    holding_cost_saving: float
    stores_included: List[str]
    correlation_matrix: np.ndarray = None


def calculate_correlation_matrix(df: pd.DataFrame, 
                                  store_ids: List[str] = None,
                                  item_id: str = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Calcula matriz de correlação de demanda entre lojas.
    
    Args:
        df: DataFrame com colunas ['date', 'store_id', 'item_id', 'demand']
        store_ids: Lista de lojas para analisar
        item_id: Item específico (se None, agrega todas as demandas)
    
    Returns:
        Tupla (DataFrame pivotado por loja, matriz de correlação)
    """
    if store_ids is None:
        store_ids = df['store_id'].unique().tolist()
    
    # Filtrar dados
    mask = df['store_id'].isin(store_ids)
    if item_id:
        mask &= df['item_id'] == item_id
    
    filtered = df[mask].copy()
    
    # Pivotar: linhas = datas, colunas = lojas
    pivot = filtered.groupby(['date', 'store_id'])['demand'].sum().unstack(fill_value=0)
    
    # Matriz de correlação
    corr_matrix = pivot.corr()
    
    return pivot, corr_matrix


def analyze_correlation_impact(corr_matrix: pd.DataFrame) -> Dict:
    """
    Analisa o impacto da correlação no potencial de pooling.
    
    Returns:
        Dict com métricas de correlação
    """
    # Extrair valores (excluindo diagonal)
    n = len(corr_matrix)
    corr_values = []
    for i in range(n):
        for j in range(i+1, n):
            corr_values.append(corr_matrix.iloc[i, j])
    
    corr_values = np.array(corr_values)
    
    return {
        'mean_correlation': np.mean(corr_values),
        'median_correlation': np.median(corr_values),
        'min_correlation': np.min(corr_values),
        'max_correlation': np.max(corr_values),
        'std_correlation': np.std(corr_values),
        'n_pairs': len(corr_values),
        'highly_correlated_pairs': np.sum(corr_values > 0.7),
        'negatively_correlated_pairs': np.sum(corr_values < 0),
        # Potencial de pooling (quanto menor a correlação média, maior o potencial)
        'pooling_potential': 1 - np.mean(corr_values)
    }


def calculate_pooled_variance(variances: np.ndarray, 
                               correlations: np.ndarray = None,
                               weights: np.ndarray = None) -> float:
    """
    Calcula variância pooled considerando correlações.
    
    Para n lojas com variâncias σ²_i e correlações ρ_ij:
    σ²_pooled = Σ σ²_i + 2 Σ Σ ρ_ij σ_i σ_j
    
    Args:
        variances: Array de variâncias por loja (σ²)
        correlations: Matriz de correlações (opcional, assume 0 se None)
        weights: Pesos por loja (opcional, assume iguais)
    
    Returns:
        Variância pooled
    """
    n = len(variances)
    stds = np.sqrt(variances)
    
    if weights is None:
        weights = np.ones(n)
    
    # Soma das variâncias ponderadas
    var_sum = np.sum(weights**2 * variances)
    
    # Soma das covariâncias (se correlações fornecidas)
    if correlations is not None:
        cov_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                cov_sum += 2 * weights[i] * weights[j] * correlations[i, j] * stds[i] * stds[j]
        var_sum += cov_sum
    
    return var_sum


def calculate_ss_reduction(df: pd.DataFrame, 
                            store_ids: List[str],
                            lead_time: float,
                            csl_target: float = 0.95,
                            h_annual: float = None) -> Dict:
    """
    Calcula redução de estoque de segurança com pooling.
    
    Compara:
    - SS descentralizado: Σ SS_i (cada loja mantém seu próprio SS)
    - SS centralizado: SS calculado na demanda agregada
    
    Returns:
        Dict com métricas de redução
    """
    z = norm.ppf(csl_target)
    
    stores_data = []
    
    for store in store_ids:
        store_df = df[df['store_id'] == store]
        daily_demand = store_df.groupby('date')['demand'].sum()
        
        mu = daily_demand.mean()
        sigma = daily_demand.std()
        sigma_L = sigma * np.sqrt(lead_time)
        ss = z * sigma_L
        
        stores_data.append({
            'store_id': store,
            'mu_daily': mu,
            'sigma_daily': sigma,
            'sigma_L': sigma_L,
            'ss': ss,
            'variance': sigma**2
        })
    
    stores_df = pd.DataFrame(stores_data)
    
    # SS Descentralizado
    ss_decentralized = stores_df['ss'].sum()
    
    # SS Centralizado (pooled)
    # Calcular correlação
    pivot, corr_matrix = calculate_correlation_matrix(df, store_ids)
    corr_np = corr_matrix.values
    variances = stores_df['variance'].values
    
    # Variância pooled
    var_pooled = calculate_pooled_variance(variances, corr_np)
    sigma_pooled = np.sqrt(var_pooled)
    sigma_L_pooled = sigma_pooled * np.sqrt(lead_time)
    ss_centralized = z * sigma_L_pooled
    
    # Redução
    ss_reduction = ss_decentralized - ss_centralized
    ss_reduction_pct = (ss_reduction / ss_decentralized) * 100 if ss_decentralized > 0 else 0
    
    # Portfolio Effect (redução teórica se correlação = 0)
    n_stores = len(store_ids)
    portfolio_effect_theoretical = 1 - 1/np.sqrt(n_stores)
    
    result = {
        'n_stores': n_stores,
        'ss_decentralized': ss_decentralized,
        'ss_centralized': ss_centralized,
        'ss_reduction': ss_reduction,
        'ss_reduction_pct': ss_reduction_pct,
        'mean_correlation': np.mean(corr_np[np.triu_indices(n_stores, k=1)]),
        'portfolio_effect_theoretical': portfolio_effect_theoretical * 100,
        'portfolio_effect_actual': ss_reduction_pct,
        'stores_detail': stores_df,
        'correlation_matrix': corr_matrix
    }
    
    if h_annual is not None:
        result['holding_cost_saving'] = ss_reduction * h_annual
    
    return result


def abc_classification(df: pd.DataFrame, 
                       value_column: str = 'demand',
                       group_by: List[str] = ['item_id']) -> pd.DataFrame:
    """
    Classifica itens em A, B, C baseado no volume/valor.
    
    Args:
        df: DataFrame com dados
        value_column: Coluna para classificação
        group_by: Colunas para agrupar
    
    Returns:
        DataFrame com classificação ABC
    """
    # Agregar valores
    agg = df.groupby(group_by)[value_column].sum().reset_index()
    agg.columns = list(group_by) + ['total_value']
    
    # Ordenar e calcular %
    agg = agg.sort_values('total_value', ascending=False).reset_index(drop=True)
    agg['cumulative_value'] = agg['total_value'].cumsum()
    agg['cumulative_pct'] = agg['cumulative_value'] / agg['total_value'].sum() * 100
    agg['value_pct'] = agg['total_value'] / agg['total_value'].sum() * 100
    
    # Classificação
    def classify(pct):
        if pct <= 80:
            return 'A'
        elif pct <= 95:
            return 'B'
        else:
            return 'C'
    
    agg['class'] = agg['cumulative_pct'].apply(classify)
    
    # Estatísticas
    agg['item_pct'] = (agg.index + 1) / len(agg) * 100
    
    return agg


def hybrid_pooling_scenario(df: pd.DataFrame,
                             abc_df: pd.DataFrame,
                             store_ids: List[str],
                             lead_time: float,
                             csl_target: float = 0.95) -> Dict:
    """
    Avalia cenário híbrido: A descentralizado, B/C centralizados.
    
    Returns:
        Dict comparando cenários
    """
    results = {}
    
    # Cenário 1: Tudo descentralizado
    all_items = abc_df['item_id'].tolist()
    ss_all_decentralized = calculate_ss_reduction(
        df[df['item_id'].isin(all_items)], 
        store_ids, lead_time, csl_target
    )['ss_decentralized']
    
    results['all_decentralized'] = {
        'ss_total': ss_all_decentralized,
        'description': 'Todas as lojas mantêm SS individual para todos os itens'
    }
    
    # Cenário 2: Tudo centralizado
    ss_all_centralized = calculate_ss_reduction(
        df[df['item_id'].isin(all_items)], 
        store_ids, lead_time, csl_target
    )['ss_centralized']
    
    results['all_centralized'] = {
        'ss_total': ss_all_centralized,
        'ss_reduction_vs_decentralized': (1 - ss_all_centralized/ss_all_decentralized) * 100,
        'description': 'Estoque centralizado para todos os itens'
    }
    
    # Cenário 3: Híbrido (A local, B/C central)
    items_a = abc_df[abc_df['class'] == 'A']['item_id'].tolist()
    items_bc = abc_df[abc_df['class'].isin(['B', 'C'])]['item_id'].tolist()
    
    if items_a and items_bc:
        # A mantém SS por loja
        ss_a_decentralized = 0
        for item in items_a:
            item_result = calculate_ss_reduction(
                df[df['item_id'] == item], 
                store_ids, lead_time, csl_target
            )
            ss_a_decentralized += item_result['ss_decentralized']
        
        # B/C centralizados
        ss_bc_centralized = calculate_ss_reduction(
            df[df['item_id'].isin(items_bc)], 
            store_ids, lead_time, csl_target
        )['ss_centralized']
        
        ss_hybrid = ss_a_decentralized + ss_bc_centralized
        
        results['hybrid_abc'] = {
            'ss_total': ss_hybrid,
            'ss_a_decentralized': ss_a_decentralized,
            'ss_bc_centralized': ss_bc_centralized,
            'ss_reduction_vs_decentralized': (1 - ss_hybrid/ss_all_decentralized) * 100,
            'n_items_a': len(items_a),
            'n_items_bc': len(items_bc),
            'description': 'Itens A descentralizados, B/C centralizados'
        }
    
    # Resumo comparativo
    results['summary'] = pd.DataFrame([
        {
            'Cenário': 'Descentralizado',
            'SS Total': results['all_decentralized']['ss_total'],
            'Redução (%)': 0.0
        },
        {
            'Cenário': 'Centralizado',
            'SS Total': results['all_centralized']['ss_total'],
            'Redução (%)': results['all_centralized']['ss_reduction_vs_decentralized']
        },
        {
            'Cenário': 'Híbrido ABC',
            'SS Total': results.get('hybrid_abc', {}).get('ss_total', 0),
            'Redução (%)': results.get('hybrid_abc', {}).get('ss_reduction_vs_decentralized', 0)
        }
    ])
    
    return results


def pooling_sensitivity_analysis(df: pd.DataFrame,
                                  store_ids: List[str],
                                  lead_time: float,
                                  csl_range: List[float] = [0.85, 0.90, 0.95, 0.98]) -> pd.DataFrame:
    """
    Análise de sensibilidade do pooling para diferentes CSLs.
    """
    results = []
    
    for csl in csl_range:
        analysis = calculate_ss_reduction(df, store_ids, lead_time, csl)
        results.append({
            'CSL': f"{csl:.0%}",
            'SS Descentralizado': analysis['ss_decentralized'],
            'SS Centralizado': analysis['ss_centralized'],
            'Redução (un)': analysis['ss_reduction'],
            'Redução (%)': analysis['ss_reduction_pct'],
            'Correlação Média': analysis['mean_correlation']
        })
    
    return pd.DataFrame(results)
