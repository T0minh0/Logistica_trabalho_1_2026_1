
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suprimir avisos de convergência
warnings.filterwarnings("ignore")

class ETSForecaster:
    def __init__(self, seasonal_periods=7, trend='add', seasonal='add'):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.model = None
        self.fit_model = None
        
    def fit(self, series):
        # Tratar zeros/negativos para modelos multiplicativos se necessário (não aqui pois usando aditivo padrão)
        try:
            self.model = ExponentialSmoothing(
                series, 
                seasonal_periods=self.seasonal_periods,
                trend=self.trend, 
                seasonal=self.seasonal
            )
            self.fit_model = self.model.fit()
        except Exception as e:
            print(f"Erro no Fit do ETS: {e}")
            self.fit_model = None

    def predict(self, horizon):
        if self.fit_model:
            return self.fit_model.forecast(horizon).values
        else:
            return np.zeros(horizon) # Fallback

class ARIMAForecaster:
    def __init__(self, order=(1,1,1), seasonal_order=(0,1,1,7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fit_model = None
        
    def fit(self, series):
        try:
            self.model = ARIMA(series, order=self.order, seasonal_order=self.seasonal_order)
            self.fit_model = self.model.fit()
        except Exception as e:
            print(f"Erro no Fit do ARIMA: {e}")
            self.fit_model = None
            
    def predict(self, horizon):
        if self.fit_model:
            return self.fit_model.forecast(steps=horizon).values
        else:
            return np.zeros(horizon)
