
import numpy as np

def calculate_pooled_variance(variances, correlation_matrix=None):
    """
    Calcula variância da soma de demandas (Pooled).
    Var(Sum) = Sum(Var) + 2*Sum(Cov)
    Se independente (corr=0), Var(Sum) = Sum(Var).
    """
    if correlation_matrix is None:
        # Assume independência
        return np.sum(variances)
    else:
        # Cálculo completo de covariância necessário se tivéssemos séries brutas
        # Por enquanto, abordagem simplista:
        # std_pool = sqrt( sum(sigma_i^2) + sum_{i!=j} rho_ij * sigma_i * sigma_j )
        pass
        
def portfolio_effect(sigma_individual_list, sigma_pooled):
    """
    Mede a redução no estoque de segurança devido ao pooling.
    SS_descentralizado ~ Sum(sigma_i)
    SS_centralizado ~ sigma_pooled
    """
    sum_individual = np.sum(sigma_individual_list)
    reduction = 1 - (sigma_pooled / sum_individual)
    return reduction
