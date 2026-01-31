
import numpy as np
from src.location.candidates import haversine_distance, CD_CANDIDATES, STORE_COORDS

def assign_stores_to_nearest_cd(store_ids, cd_names=None):
    """
    Atribui cada loja ao CD candidato geograficamente mais próximo.
    """
    if cd_names is None:
        cd_names = list(CD_CANDIDATES.keys())
        
    assignments = {}
    
    for store_id in store_ids:
        store_loc = STORE_COORDS.get(store_id)
        if not store_loc:
            assignments[store_id] = None
            continue
            
        min_dist = float('inf')
        best_cd = None
        
        for cd in cd_names:
            cd_loc = CD_CANDIDATES[cd]
            dist = haversine_distance(store_loc, cd_loc)
            if dist < min_dist:
                min_dist = dist
                best_cd = cd
        
        assignments[store_id] = {'cd': best_cd, 'distance_km': min_dist}
        
    return assignments
