
import numpy as np
from scipy.stats import norm

class RQPolicy:
    """
    Política (R, Q) sob demanda estocástica.
    Q é geralmente fixo (ex: EOQ) ou otimizado simultaneamente.
    R = mu_L + SS
    SS = z * sigma_L
    """
    def __init__(self, lead_time_days, csl_target=0.95):
        self.L = lead_time_days
        self.csl_target = csl_target
        
    def calculate_parameters(self, forecast_mean_daily, forecast_sigma_daily):
        """
        Calcula o Ponto de Ressuprimento (R) e Estoque de Segurança (SS).
        Assume demanda diária independente (aproximação).
        mu_L = mu * L
        sigma_L = sigma * sqrt(L)
        """
        mu_L = forecast_mean_daily * self.L
        sigma_L = forecast_sigma_daily * np.sqrt(self.L)
        
        z_score = norm.ppf(self.csl_target)
        ss = z_score * sigma_L
        
        R = mu_L + ss
        
        return {
            'R': R,
            'SS': ss,
            'mu_L': mu_L,
            'sigma_L': sigma_L,
            'z': z_score
        }
