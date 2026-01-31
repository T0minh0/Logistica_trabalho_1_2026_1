
import numpy as np
import pandas as pd

# Dados M5 são anonimizados, então usamos coordenadas proxy ou reais se assumirmos
# WI (Wisconsin), CA (Califórnia), TX (Texas).
# Vamos mapear lojas para latitude/longitude aproximada.
# CA: Califórnia (Sacramento/LA?)
# TX: Texas (Dallas/Austin?)
# WI: Wisconsin (Milwaukee?)

STORE_COORDS = {
    'CA_1': (34.05, -118.24), # LA Proxy
    'CA_2': (34.05, -117.24), # Perto de LA
    'CA_3': (38.58, -121.49), # Sacramento
    'CA_4': (37.77, -122.41), # SF
    'TX_1': (32.77, -96.79),  # Dallas
    'TX_2': (32.77, -97.79),  # Perto de Dallas
    'TX_3': (29.76, -95.36),  # Houston
    'WI_1': (43.03, -87.91),  # Milwaukee
    'WI_2': (43.07, -89.40),  # Madison
    'WI_3': (44.51, -88.01)   # Green Bay
}

# Locais de CD Potenciais (Candidatos)
CD_CANDIDATES = {
    'CD_West': (36.77, -119.41), # California Central (Fresno)
    'CD_Central': (31.96, -99.90), # Texas Central
    'CD_North': (44.50, -89.50)    # Wisconsin Central
}

def get_store_coordinates(store_ids):
    """Retorna uma lista de coordenadas para os IDs de loja fornecidos."""
    coords = []
    for sid in store_ids:
        coords.append(STORE_COORDS.get(sid, (0,0)))
    return np.array(coords)

def haversine_distance(coord1, coord2):
    """
    Distância aproximada em km entre duas tuplas de lat/long.
    """
    R = 6371  # Raio da Terra em km
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c
