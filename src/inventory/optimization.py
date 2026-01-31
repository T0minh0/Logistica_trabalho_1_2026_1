"""
Otimização de Estoques Multi-Item
Busca Q* ótimo considerando custo de ruptura e restrições de orçamento.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


@dataclass
class ItemParams:
    """Parâmetros de um item para otimização."""
    item_id: str
    demand_mean: float      # Demanda média diária
    demand_std: float       # Desvio padrão diário
    unit_cost: float        # Custo unitário
    lead_time: float        # Lead time (dias)
    lead_time_std: float = 0.0  # Desvio padrão do lead time
    priority: float = 1.0   # Prioridade (opcional)


def optimal_Q_with_stockout(D: float, K: float, h: float, p: float,
                             sigma: float, L: float, csl: float = None) -> Dict:
    """
    Calcula Q* ótimo considerando custo de ruptura.
    
    Modelo EOQ estendido com backorder:
    Q* = √(2DK/h × (h + p)/p)
    
    Args:
        D: Demanda anual
        K: Custo por pedido
        h: Custo de holding por unidade/ano
        p: Custo de falta por unidade
        sigma: Desvio padrão da demanda diária
        L: Lead time (dias)
        csl: CSL alvo (opcional, calcula baseado em p se None)
    
    Returns:
        Dict com Q*, R, SS e custos
    """
    # EOQ com backorder
    Q_star = np.sqrt(2 * D * K / h) * np.sqrt((h + p) / p)
    
    # Nível de serviço ótimo (se não especificado)
    if csl is None:
        # Fórmula: CSL* = 1 - (h × Q) / (p × D)
        csl_optimal = 1 - (h * Q_star) / (p * D)
        csl_optimal = max(0.5, min(0.999, csl_optimal))
    else:
        csl_optimal = csl
    
    z = norm.ppf(csl_optimal)
    
    # Demanda durante lead time
    sigma_L = sigma * np.sqrt(L)
    mu_L = (D / 365) * L
    
    # Estoque de segurança e ponto de ressuprimento
    SS = z * sigma_L
    R = mu_L + SS
    
    # Custos
    ordering_cost = K * D / Q_star
    holding_cost = h * (Q_star / 2 + SS)
    
    # Custo esperado de falta
    # E[shortage] ≈ σ_L × L(z) onde L(z) = φ(z) - z(1-Φ(z))
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    loss_function = pdf_z - z * (1 - cdf_z)
    expected_shortage = sigma_L * loss_function
    stockout_cost = p * (D / Q_star) * expected_shortage
    
    total_cost = ordering_cost + holding_cost + stockout_cost
    
    return {
        'Q_star': Q_star,
        'R': R,
        'SS': SS,
        'csl_optimal': csl_optimal,
        'z_score': z,
        'ordering_cost': ordering_cost,
        'holding_cost': holding_cost,
        'stockout_cost': stockout_cost,
        'total_cost': total_cost,
        'fill_rate': 1 - expected_shortage * (D / Q_star) / D
    }


def sensitivity_analysis_stockout_cost(D: float, K: float, h: float,
                                         sigma: float, L: float,
                                         p_range: List[float] = None) -> pd.DataFrame:
    """
    Análise de sensibilidade ao custo de falta.
    """
    if p_range is None:
        p_range = [h * i for i in [1, 2, 5, 10, 20, 50, 100]]
    
    results = []
    for p in p_range:
        opt = optimal_Q_with_stockout(D, K, h, p, sigma, L)
        results.append({
            'Custo Falta (p)': p,
            'p/h Ratio': p / h,
            'Q*': opt['Q_star'],
            'SS': opt['SS'],
            'R': opt['R'],
            'CSL Ótimo (%)': opt['csl_optimal'] * 100,
            'Custo Total': opt['total_cost']
        })
    
    return pd.DataFrame(results)


def multi_item_budget_constraint(items: List[ItemParams],
                                  K: float, h_pct: float, p: float,
                                  budget: float, csl_target: float = 0.95) -> Dict:
    """
    Otimização multi-item com restrição de orçamento.
    
    max Σ Fill_Rate_i × Priority_i
    s.t. Σ (unit_cost_i × Q_i / 2 + unit_cost_i × SS_i) ≤ Budget
    
    Args:
        items: Lista de ItemParams
        K: Custo por pedido (comum)
        h_pct: Holding cost como % do valor
        p: Custo de falta por unidade
        budget: Orçamento máximo para estoque
        csl_target: CSL alvo
    
    Returns:
        Dict com alocação ótima
    """
    z = norm.ppf(csl_target)
    n = len(items)
    
    # Calcular parâmetros por item
    results = []
    for item in items:
        D = item.demand_mean * 365
        h = h_pct * item.unit_cost
        sigma_L = item.demand_std * np.sqrt(item.lead_time)
        
        # EOQ
        Q_eoq = np.sqrt(2 * D * K / h)
        SS = z * sigma_L
        
        # Custo de estoque
        avg_inventory = Q_eoq / 2 + SS
        inventory_value = avg_inventory * item.unit_cost
        
        results.append({
            'item_id': item.item_id,
            'D_annual': D,
            'Q_eoq': Q_eoq,
            'SS': SS,
            'avg_inventory': avg_inventory,
            'inventory_value': inventory_value,
            'priority': item.priority,
            'unit_cost': item.unit_cost
        })
    
    df = pd.DataFrame(results)
    
    # Verificar se cabe no orçamento
    total_value = df['inventory_value'].sum()
    
    if total_value <= budget:
        # Cabe! Retornar solução irrestrita
        df['allocation_pct'] = 100.0
        df['Q_allocated'] = df['Q_eoq']
        df['SS_allocated'] = df['SS']
        
        return {
            'status': 'feasible_unconstrained',
            'total_inventory_value': total_value,
            'budget': budget,
            'budget_utilization': total_value / budget * 100,
            'allocation': df
        }
    
    # Precisa otimizar - usar heurística de alocação proporcional
    # Prioridade: itens com maior (priority × D / costo) recebem mais
    df['allocation_score'] = df['priority'] * df['D_annual'] / df['unit_cost']
    df['allocation_weight'] = df['allocation_score'] / df['allocation_score'].sum()
    
    # Alocar orçamento proporcionalmente
    df['budget_allocated'] = budget * df['allocation_weight']
    
    # Calcular Q e SS com orçamento limitado
    def optimize_q_ss(row):
        # Dado budget = Q/2 × c + SS × c
        # Q/2 + SS = budget / c
        target_inventory = row['budget_allocated'] / row['unit_cost']
        
        # Manter proporção original
        if row['avg_inventory'] > 0:
            scale = target_inventory / row['avg_inventory']
        else:
            scale = 1.0
        
        return pd.Series({
            'Q_allocated': row['Q_eoq'] * scale,
            'SS_allocated': row['SS'] * scale,
            'allocated_inventory': row['Q_allocated'] / 2 + row['SS_allocated']
        })
    
    allocation = df.apply(optimize_q_ss, axis=1)
    df = pd.concat([df, allocation], axis=1)
    df['allocation_pct'] = df['budget_allocated'] / df['inventory_value'] * 100
    
    return {
        'status': 'constrained_optimized',
        'total_inventory_value': budget,
        'original_requirement': total_value,
        'budget': budget,
        'budget_utilization': 100.0,
        'savings_required': total_value - budget,
        'allocation': df
    }


def newsvendor_optimal_Q(D_mean: float, D_std: float, 
                          unit_cost: float, selling_price: float,
                          salvage_value: float = 0) -> Dict:
    """
    Modelo Newsvendor para perecíveis/sazonais.
    
    Q* = F^(-1)(Cu / (Cu + Co))
    
    Args:
        D_mean: Demanda média
        D_std: Desvio padrão da demanda
        unit_cost: Custo de compra
        selling_price: Preço de venda
        salvage_value: Valor de salvamento
    
    Returns:
        Dict com Q* e análise
    """
    # Custos marginais
    Cu = selling_price - unit_cost  # Custo de subprodução (underage)
    Co = unit_cost - salvage_value   # Custo de superprodução (overage)
    
    # Critical ratio
    critical_ratio = Cu / (Cu + Co)
    
    # Quantidade ótima
    z = norm.ppf(critical_ratio)
    Q_star = D_mean + z * D_std
    
    # Métricas
    expected_sales = D_mean * norm.cdf(z) + D_std * norm.pdf(z)
    expected_leftover = Q_star - expected_sales
    expected_shortage = D_mean - expected_sales
    
    expected_profit = (
        selling_price * expected_sales 
        + salvage_value * expected_leftover 
        - unit_cost * Q_star
    )
    
    return {
        'Q_star': max(0, Q_star),
        'critical_ratio': critical_ratio,
        'z_score': z,
        'expected_sales': expected_sales,
        'expected_leftover': expected_leftover,
        'expected_shortage': expected_shortage,
        'expected_profit': expected_profit,
        'fill_rate': expected_sales / D_mean if D_mean > 0 else 0
    }


def joint_ordering_optimization(items: List[ItemParams],
                                 K_joint: float, K_individual: float,
                                 h_pct: float) -> Dict:
    """
    Otimização de pedidos conjuntos (Power-of-Two).
    
    Compara:
    - Pedidos individuais por item
    - Pedido conjunto com período comum
    
    Args:
        items: Lista de itens
        K_joint: Custo fixo de pedido conjunto
        K_individual: Custo de adicionar item ao pedido
        h_pct: Holding cost %
    
    Returns:
        Dict com comparação
    """
    results_individual = []
    results_joint = []
    
    # Custo individual
    for item in items:
        D = item.demand_mean * 365
        h = h_pct * item.unit_cost
        K = K_joint + K_individual  # Custo total por pedido individual
        
        Q = np.sqrt(2 * D * K / h)
        cost = np.sqrt(2 * D * K * h)
        
        results_individual.append({
            'item_id': item.item_id,
            'Q': Q,
            'n_orders': D / Q,
            'annual_cost': cost
        })
    
    df_ind = pd.DataFrame(results_individual)
    total_individual = df_ind['annual_cost'].sum()
    
    # Custo conjunto - encontrar T* comum
    # CT(T) = K_joint/T + Σ K_individual/T + Σ h_i × D_i × T / 2
    
    def joint_cost(T):
        fixed_cost = K_joint / T
        item_cost = K_individual * len(items) / T
        
        holding = 0
        for item in items:
            D = item.demand_mean * 365
            h = h_pct * item.unit_cost
            holding += h * D * T / 2
        
        return fixed_cost + item_cost + holding
    
    # Otimizar T
    res = minimize_scalar(joint_cost, bounds=(0.01, 1), method='bounded')
    T_star = res.x
    
    for item in items:
        D = item.demand_mean * 365
        Q_joint = D * T_star
        
        results_joint.append({
            'item_id': item.item_id,
            'Q': Q_joint,
            'T_star': T_star,
            'n_orders': 1 / T_star
        })
    
    df_joint = pd.DataFrame(results_joint)
    total_joint = joint_cost(T_star)
    
    return {
        'individual': {
            'allocation': df_ind,
            'total_cost': total_individual,
            'avg_orders_per_item': df_ind['n_orders'].mean()
        },
        'joint': {
            'allocation': df_joint,
            'total_cost': total_joint,
            'T_star_years': T_star,
            'T_star_days': T_star * 365,
            'orders_per_year': 1 / T_star
        },
        'savings': {
            'absolute': total_individual - total_joint,
            'percentage': (total_individual - total_joint) / total_individual * 100
        }
    }


if HAS_PULP:
    def multi_item_milp_optimization(items: List[ItemParams],
                                      K: float, h_pct: float,
                                      budget: float, 
                                      min_fill_rate: float = 0.90) -> Dict:
        """
        Otimização MILP para multi-item com orçamento.
        
        Usa PuLP para programação linear inteira mista.
        """
        prob = LpProblem("MultiItem_Inventory", LpMinimize)
        
        # Variáveis
        Q = {i: LpVariable(f"Q_{items[i].item_id}", lowBound=1) for i in range(len(items))}
        SS = {i: LpVariable(f"SS_{items[i].item_id}", lowBound=0) for i in range(len(items))}
        
        # Função objetivo: minimizar custo total
        total_cost = lpSum([
            K * (items[i].demand_mean * 365) / Q[i] +  # Ordering cost
            h_pct * items[i].unit_cost * (Q[i] / 2 + SS[i])  # Holding cost
            for i in range(len(items))
        ])
        prob += total_cost
        
        # Restrição de orçamento
        prob += lpSum([
            items[i].unit_cost * (Q[i] / 2 + SS[i])
            for i in range(len(items))
        ]) <= budget
        
        # Restrições de fill rate (aproximação linear)
        z_min = norm.ppf(min_fill_rate)
        for i in range(len(items)):
            sigma_L = items[i].demand_std * np.sqrt(items[i].lead_time)
            prob += SS[i] >= z_min * sigma_L
        
        # Resolver
        prob.solve()
        
        # Extrair resultados
        results = []
        for i in range(len(items)):
            results.append({
                'item_id': items[i].item_id,
                'Q': value(Q[i]),
                'SS': value(SS[i]),
                'avg_inventory': value(Q[i]) / 2 + value(SS[i]),
                'inventory_value': items[i].unit_cost * (value(Q[i]) / 2 + value(SS[i]))
            })
        
        return {
            'status': LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'allocation': pd.DataFrame(results)
        }
