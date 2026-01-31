
import numpy as np
import pandas as pd

def analyze_intermittency(df):
    """
    Analisa características de intermitência:
    - Percentual de demanda zero
    - Intervalo médio entre demandas (ADI)
    - CV²
    """
    def _calc_intermittency(series):
        # Remover zeros iniciais? (Opcional, mas aqui pegamos a série completa)
        # Calcular ADI
        demand_idx = np.where(series > 0)[0]
        if len(demand_idx) < 2:
            adi = len(series) # Fallback
        else:
            intervals = np.diff(demand_idx)
            adi = np.mean(intervals)
            
        # CV2
        mean = series.mean()
        std = series.std()
        cv2 = (std/mean)**2 if mean > 0 else np.nan
        
        # Zero pct
        zero_pct = (series == 0).mean()
        
        return pd.Series({
            'adi': adi,
            'cv2': cv2,
            'zero_pct': zero_pct
        })

    results = df.groupby(['store_id', 'item_id'])['demand'].apply(_calc_intermittency).unstack()
    
    # Classificação
    # Classificação SBC:
    # Smooth (Suave): ADI < 1.32 e CV2 < 0.49
    # Intermittent (Intermitente): ADI >= 1.32 e CV2 < 0.49
    # Erratic (Errático): ADI < 1.32 e CV2 >= 0.49
    # Lumpy (Grumoso): ADI >= 1.32 e CV2 >= 0.49
    
    def classify(row):
        adi, cv2 = row['adi'], row['cv2']
        if pd.isna(adi) or pd.isna(cv2): return 'Desconhecido'
        if adi < 1.32 and cv2 < 0.49: return 'Suave'
        if adi >= 1.32 and cv2 < 0.49: return 'Intermitente'
        if adi < 1.32 and cv2 >= 0.49: return 'Errático'
        return 'Grumoso'

    results['classification'] = results.apply(classify, axis=1)
    return results.reset_index()
