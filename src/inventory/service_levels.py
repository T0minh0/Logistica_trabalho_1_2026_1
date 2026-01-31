
import numpy as np
from scipy.stats import norm

def expected_shortage_per_cycle(Q, R, sigma_L):
    """
    Falta esperada por ciclo de ressuprimento (E[n(R)]) para demanda Normal.
    Função de Perda Padrão L(z).
    """
    z = (R - (R - 0)) / sigma_L # Simplificado se R é centrado? Não.
    # z = (R - mu_L) / sigma_L
    # Mas argumentos passados geralmente não implicam mu_L diretamente aqui se calculamos SS separadamente?
    # Vamos refrasear: fórmula padrão usa função de perda normal padrão.
    pass 

def calculate_fill_rate(Q, R, mu_L, sigma_L):
    """
    Fill Rate (beta) = 1 - (Falta Esperada / Q)
    """
    z = (R - mu_L) / sigma_L
    
    # Função de Perda Normal Padrão L(z) = pdf(z) - z*(1-cdf(z))
    L_z = norm.pdf(z) - z * (1 - norm.cdf(z))
    
    expected_shortage = sigma_L * L_z
    
    fill_rate = 1 - (expected_shortage / Q)
    return fill_rate

def implied_service_level(p, h, D, Q):
    """
    CSL Ótimo dado custo de penalidade p.
    Tipo jornaleiro (Newsvendor) ou otimização (p, h): CSL* = p / (p + h) ? Não para revisão contínua (R,Q).
    
    Aproximação para (R,Q): F(z) = 1 - (Q*h)/(D*p)
    """
    cdf_z = 1 - (Q * h) / (D * p)
    return cdf_z
