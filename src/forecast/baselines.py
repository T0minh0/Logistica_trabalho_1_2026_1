
import numpy as np
import pandas as pd

class BaselineForecaster:
    def __init__(self, method='naive', seasonality=7, window=7):
        """
        method: 'naive', 'seasonal_naive', ou 'moving_average'
        seasonality: período sazonal (para seasonal_naive)
        window: janela para média móvel
        """
        self.method = method
        self.seasonality = seasonality
        self.window = window
        self.last_value = None
        self.history = None
        
    def fit(self, series):
        self.history = series
        if self.method == 'naive':
            self.last_value = series.iloc[-1]
        elif self.method == 'seasonal_naive':
            pass # Não precisa de fit, usa histórico no predict
        elif self.method == 'moving_average':
            pass # Usa histórico
            
    def predict(self, horizon):
        if self.method == 'naive':
            return np.full(horizon, self.last_value)
        
        elif self.method == 'seasonal_naive':
            # Repetir último ciclo
            seasonal_cycle = self.history.iloc[-self.seasonality:].values
            forecast = []
            for i in range(horizon):
                forecast.append(seasonal_cycle[i % self.seasonality])
            return np.array(forecast)
            
        elif self.method == 'moving_average':
            # Previsão de média móvel simples (constante)
            ma = self.history.iloc[-self.window:].mean()
            return np.full(horizon, ma)
            
    def __repr__(self):
        return f"BaselineForecaster(method='{self.method}')"
