
import pandas as pd
import numpy as np
from src.config import RAW_SALES, RAW_CALENDAR, RAW_PRICES

def load_m5_data():
    """
    Carrega o dataset M5 Forecasting (vendas, calendário, preços).
    Retorna:
        sales (pd.DataFrame): Histórico de vendas (formato wide)
        calendar (pd.DataFrame): Features de calendário
        prices (pd.DataFrame): Preços de venda
    """
    print("Carregando dados de vendas...")
    sales = pd.read_csv(RAW_SALES)
    
    print("Carregando dados de calendário...")
    calendar = pd.read_csv(RAW_CALENDAR)
    
    print("Carregando dados de preços...")
    prices = pd.read_csv(RAW_PRICES)
    
    return sales, calendar, prices

if __name__ == "__main__":
    # Teste rápido
    s, c, p = load_m5_data()
    print(f"Shape Vendas: {s.shape}")
    print(f"Shape Calendário: {c.shape}")
    print(f"Shape Preços: {p.shape}")
