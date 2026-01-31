
import pandas as pd
import numpy as np
import statsmodels.api as sm

class ARIMAXForecaster:
    """
    ARIMA com variáveis eXógenas (como preço, eventos).
    Wrapper simplificado em torno do statsmodels SARIMAX.
    """
    def __init__(self, order=(1,0,0), seasonal_order=(0,0,0,0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fit_model = None
        
    def fit(self, series, exog):
        """
        series: demanda alvo
        exog: DataFrame pandas de features exógenas (alinhado com o índice da série)
        """
        try:
            # Garantir que exog é numérico e tratar NaNs
            exog = exog.fillna(0).astype(float)
            
            self.model = sm.tsa.statespace.SARIMAX(
                series, 
                exog=exog,
                order=self.order, 
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fit_model = self.model.fit(disp=False)
        except Exception as e:
            print(f"Erro no Fit do ARIMAX: {e}")
            self.fit_model = None
            
    def predict(self, horizon, exog_future):
        """
        exog_future: features exógenas para o horizonte de previsão
        """
        if self.fit_model:
            exog_future = exog_future.fillna(0).astype(float)
            return self.fit_model.forecast(steps=horizon, exog=exog_future).values
        else:
            return np.zeros(horizon)
