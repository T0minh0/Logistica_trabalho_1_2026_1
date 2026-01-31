
import pandas as pd
import numpy as np

def compute_statistics(df, group_cols=['store_id', 'item_id']):
    """
    Calcula estatísticas descritivas básicas para a demanda por grupo.
    """
    stats = df.groupby(group_cols)['demand'].agg(
        mean='mean',
        std='std',
        min='min',
        max='max',
        count='count'
    )
    stats['cv'] = stats['std'] / stats['mean']
    return stats.reset_index()

def correlation_analysis(df):
    """
    Calcula a correlação entre lojas para os mesmos itens.
    Requer que df esteja em formato long com múltiplas lojas.
    """
    # Pivotar para obter lojas como colunas, index = data + item
    pivot_df = df.pivot_table(index=['date', 'item_id'], 
                              columns='store_id', 
                              values='demand').reset_index()
    
    # Calcular matriz de correlação
    corr_matrix = pivot_df.drop(columns=['date', 'item_id']).corr()
    return corr_matrix
