
import numpy as np

def mae(y_true, y_pred):
    """Erro Médio Absoluto"""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Raiz do Erro Quadrático Médio"""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error (Erro Percentual Absoluto Ponderado)"""
    if np.sum(y_true) == 0:
        return np.inf
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (0-200%)"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)

def evaluate_forecast(y_true, y_pred):
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'WAPE': wape(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred)
    }
