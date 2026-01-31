
import pandas as pd
import numpy as np
from src.config import SELECTED_STORE_IDS, SELECTED_CATEGORIES, TOP_N_ITEMS

def melt_sales(sales_df):
    """
    Converte o DataFrame de vendas do formato wide para long.
    """
    # Identificar colunas d_
    d_cols = [c for c in sales_df.columns if c.startswith('d_')]
    
    # Variáveis de ID
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    # Melt
    print("Realizando melt no dataframe de vendas (isso pode demorar um pouco)...")
    df_long = pd.melt(sales_df, 
                      id_vars=id_vars, 
                      value_vars=d_cols, 
                      var_name='d', 
                      value_name='demand')
    return df_long

def preprocess_data(sales, calendar, prices):
    """
    Pipeline completo de pré-processamento:
    1. Filtrar por lojas/categorias selecionadas (Recorte)
    2. Selecionar Top N itens por volume
    3. Melt para formato long
    4. Merge com calendário e preços
    5. Engenharia de Features
    """
    
    # 1. Filtrar Recorte
    print(f"Filtrando por lojas: {SELECTED_STORE_IDS} e categorias: {SELECTED_CATEGORIES}")
    sales_filtered = sales[
        (sales['store_id'].isin(SELECTED_STORE_IDS)) & 
        (sales['cat_id'].isin(SELECTED_CATEGORIES))
    ].copy()
    
    # 2. Selecionar Top N itens (para manter o dataset gerenciável para o escopo deste projeto)
    # Calcular demanda total por item para encontrar os top N
    d_cols = [c for c in sales_filtered.columns if c.startswith('d_')]
    sales_filtered['total_volume'] = sales_filtered[d_cols].sum(axis=1)
    
    top_items = sales_filtered.nlargest(TOP_N_ITEMS, 'total_volume')['item_id'].unique()
    
    print(f"Selecionando top {TOP_N_ITEMS} itens por volume...")
    sales_final = sales_filtered[sales_filtered['item_id'].isin(top_items)].copy()
    sales_final = sales_final.drop(columns=['total_volume'])
    
    # 3. Melt
    df = melt_sales(sales_final)
    
    # 4. Merges
    print("Realizando merge com calendário...")
    df = df.merge(calendar, on='d', how='left')
    
    print("Realizando merge com preços...")
    df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    
    # 5. Engenharia de Features
    print("Criando features...")
    
    # Features de Data
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Eventos
    df['is_event'] = (~df['event_name_1'].isna()).astype(int)
    
    # Preencher NAs de preço (se houver) com forward fill por grupo (opcional, mas boa prática)
    # Por simplicidade aqui removemos linhas onde o preço está faltando (geralmente significa que o item não foi vendido ainda)
    df = df.dropna(subset=['sell_price'])
    
    print(f"Shape final do DataFrame: {df.shape}")
    return df

if __name__ == "__main__":
    from src.io_load import load_m5_data
    s, c, p = load_m5_data()
    df = preprocess_data(s, c, p)
    print(df.head())
