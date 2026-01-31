
from pathlib import Path

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Dados"
RESULTS_DIR = BASE_DIR / "Resultados"

RAW_SALES = DATA_DIR / "sales_train_validation.csv"
RAW_CALENDAR = DATA_DIR / "calendar.csv"
RAW_PRICES = DATA_DIR / "sell_prices.csv"

# Recorte (Subseção dos dados para processar)
# CA_1 é uma loja na Califórnia. 
# Vamos focar em 'FOODS' e 'HOBBIES' conforme sugestões/recortes do documento.
SELECTED_STORE_IDS = ['CA_1', 'CA_2'] 
SELECTED_CATEGORIES = ['FOODS', 'HOBBIES']
# Número de itens principais a selecionar para análise detalhada
TOP_N_ITEMS = 30 

# Parâmetros Econômicos (Cenários)
# Cenário 1: Base
PARAMS_SCENARIO_1 = {
    'K': 50.0,      # Custo de pedido (Ordering cost)
    'h_pct': 0.20,  # Percentual de custo de manutenção por ano (20% do custo unitário)
    'p_factor': 5.0 # Fator de penalidade por falta de estoque (5x custo de manutenção por dia, proxy aproximada)
}

# Níveis de Serviço
TARGET_CSL = 0.95
TARGET_FILL_RATE = 0.98

# Previsão
TRAIN_END_DAY = 1913 - 28 # Conjunto de validação M5 é de 28 dias
FORECAST_HORIZON = 28
