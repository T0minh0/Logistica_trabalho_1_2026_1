
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sales_series(df, item_id, store_id, title=None):
    """
    Plota a série temporal de demanda para uma combinação específica de item-loja.
    """
    subset = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].sort_values('date')
    
    plt.figure(figsize=(12, 6))
    plt.plot(subset['date'], subset['demand'], label='Demanda')
    plt.title(title or f'Vendas para {item_id} em {store_id}')
    plt.xlabel('Data')
    plt.ylabel('Demanda Diária')
    plt.legend()
    plt.tight_layout()
    return plt

def plot_seasonality_heatmap(df, category):
    """
    Plota um heatmap da demanda média por Dia da Semana x Mês.
    """
    subset = df[df['cat_id'] == category].copy()
    heatmap_data = subset.groupby(['day_of_week', 'month'])['demand'].mean().unstack()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f'Heatmap de Sazonalidade - {category} (Demanda Média)')
    plt.xlabel('Mês')
    plt.ylabel('Dia da Semana (0=Seg)')
    plt.tight_layout()
    return plt
