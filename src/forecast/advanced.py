"""
Modelos Avançados de Previsão - LightGBM e Ensemble
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

class LagFeatureForecaster:
    """
    Modelo baseado em features de lag + LightGBM para previsão.
    Muito mais preciso que ETS/ARIMA para séries com padrões complexos.
    """
    def __init__(self, lags=[1, 7, 14, 28], window_sizes=[7, 14, 28]):
        self.lags = lags
        self.window_sizes = window_sizes
        self.model = None
        self.scaler = StandardScaler()
        self.last_values = None
        
    def _create_features(self, series, is_train=True):
        """Cria features de lag e estatísticas de janela."""
        df = pd.DataFrame({'demand': series.values})
        
        # Features de Lag
        for lag in self.lags:
            df[f'lag_{lag}'] = df['demand'].shift(lag)
        
        # Médias Móveis
        for window in self.window_sizes:
            df[f'rolling_mean_{window}'] = df['demand'].shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = df['demand'].shift(1).rolling(window).std()
            df[f'rolling_min_{window}'] = df['demand'].shift(1).rolling(window).min()
            df[f'rolling_max_{window}'] = df['demand'].shift(1).rolling(window).max()
        
        # Features de tendência
        df['diff_1'] = df['demand'].diff(1)
        df['diff_7'] = df['demand'].diff(7)
        
        # Features cíclicas (dia da semana aproximado)
        df['position'] = np.arange(len(df))
        df['day_of_week'] = df['position'] % 7
        df['week_of_month'] = (df['position'] // 7) % 4
        
        # Remover linhas com NaN
        if is_train:
            df = df.dropna()
        
        return df
    
    def fit(self, series):
        """Treina o modelo LightGBM."""
        df = self._create_features(series)
        
        X = df.drop(columns=['demand', 'position'])
        y = df['demand']
        
        # Salvar últimos valores para previsão
        self.last_values = series.copy()
        
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            self.model.fit(X, y)
        else:
            # Fallback para sklearn
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
            self.model.fit(X, y)
            
    def predict(self, horizon):
        """Faz previsões passo a passo (recursive forecasting)."""
        predictions = []
        current_series = self.last_values.copy()
        
        for h in range(horizon):
            # Criar features para o próximo passo
            df = self._create_features(current_series, is_train=False)
            
            # Pegar última linha (sem NaN)
            last_row = df.iloc[-1:].copy()
            last_row = last_row.drop(columns=['demand', 'position'])
            last_row = last_row.fillna(0)
            
            # Prever
            pred = self.model.predict(last_row)[0]
            pred = max(0, pred)  # Demanda não pode ser negativa
            predictions.append(pred)
            
            # Atualizar série para próxima iteração
            new_idx = len(current_series)
            current_series = pd.concat([
                current_series, 
                pd.Series([pred], index=[new_idx])
            ])
        
        return np.array(predictions)


class EnsembleForecaster:
    """
    Ensemble de múltiplos modelos para previsão robusta.
    Combina ETS, ARIMA e LightGBM.
    """
    def __init__(self, weights=None):
        self.weights = weights or {'ets': 0.25, 'arima': 0.25, 'lgbm': 0.50}
        self.models = {}
        self.fitted = False
        
    def fit(self, series):
        """Treina todos os modelos do ensemble."""
        from src.forecast.ets_arima import ETSForecaster, ARIMAForecaster
        
        # ETS
        try:
            self.models['ets'] = ETSForecaster(seasonal_periods=7)
            self.models['ets'].fit(series)
        except:
            self.models['ets'] = None
            
        # ARIMA
        try:
            self.models['arima'] = ARIMAForecaster(order=(1,1,1), seasonal_order=(1,0,1,7))
            self.models['arima'].fit(series)
        except:
            self.models['arima'] = None
            
        # LightGBM
        try:
            self.models['lgbm'] = LagFeatureForecaster()
            self.models['lgbm'].fit(series)
        except:
            self.models['lgbm'] = None
            
        self.fitted = True
        
    def predict(self, horizon):
        """Combina previsões de todos os modelos."""
        predictions = {}
        
        for name, model in self.models.items():
            if model is not None:
                try:
                    predictions[name] = model.predict(horizon)
                except:
                    predictions[name] = None
        
        # Combinar previsões com pesos
        valid_preds = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_preds:
            return np.zeros(horizon)
        
        # Normalizar pesos
        total_weight = sum(self.weights[k] for k in valid_preds.keys())
        
        combined = np.zeros(horizon)
        for name, pred in valid_preds.items():
            weight = self.weights[name] / total_weight
            combined += weight * pred
            
        return combined


class SeasonalDecompForecaster:
    """
    Previsão baseada em decomposição sazonal + tendência.
    """
    def __init__(self, period=7):
        self.period = period
        self.trend_model = None
        self.seasonal_pattern = None
        self.last_level = None
        
    def fit(self, series):
        """Decompõe a série e modela tendência."""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Decomposição
        decomp = seasonal_decompose(series, model='additive', period=self.period, extrapolate_trend='freq')
        
        # Salvar padrão sazonal (último ciclo)
        self.seasonal_pattern = decomp.seasonal.tail(self.period).values
        
        # Modelar tendência com regressão linear simples
        trend = decomp.trend.dropna()
        X = np.arange(len(trend)).reshape(-1, 1)
        y = trend.values
        
        from sklearn.linear_model import LinearRegression
        self.trend_model = LinearRegression()
        self.trend_model.fit(X, y)
        
        # Nível atual
        self.last_level = len(series)
        self.last_trend_value = self.trend_model.predict([[len(trend) - 1]])[0]
        
    def predict(self, horizon):
        """Projeta tendência + sazonalidade."""
        predictions = []
        
        for h in range(horizon):
            # Tendência futura
            trend_pred = self.trend_model.predict([[self.last_level + h]])[0]
            
            # Sazonalidade (cíclica)
            seasonal_idx = h % self.period
            seasonal_pred = self.seasonal_pattern[seasonal_idx]
            
            pred = trend_pred + seasonal_pred
            predictions.append(max(0, pred))
            
        return np.array(predictions)
