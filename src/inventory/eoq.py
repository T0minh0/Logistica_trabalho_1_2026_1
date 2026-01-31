
import numpy as np

def calculate_eoq(D, K, h):
    """
    Economic Order Quantity (Lote Econômico de Compra - Determinístico)
    D: Demanda Anual (unidades)
    K: Custo de Pedido (custo fixo por pedido)
    h: Custo de Manutenção por unidade por ano
    """
    if h <= 0 or D <= 0:
        return 0
    return np.sqrt((2 * D * K) / h)

def total_cost_deterministic(Q, D, K, h):
    """
    Custo Total Determinístico = (Custo de Pedido) + (Custo de Manutenção)
    """
    if Q <= 0: return np.inf
    ordering_cost = (D / Q) * K
    holding_cost = (Q / 2) * h
    return ordering_cost + holding_cost
