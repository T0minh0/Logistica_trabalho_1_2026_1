
import numpy as np

def calculate_transport_cost(distance_km, volume, cost_per_km_unit=0.01, fixed_cost=100):
    """
    Função proxy para custo de transporte.
    Custo = Fixo + (Taxa Variável * Distância * Volume)
    
    Nota: Idealmente volume deveria ser em pallets ou m3. Aqui 'volume' é unidades vendidas.
    Assumimos 1 unidade ~ fração constante de uma remessa.
    """
    variable_cost = cost_per_km_unit * distance_km * volume
    total = fixed_cost + variable_cost
    return total

def estimate_network_transport_cost(assignments, store_volumes):
    """
    assignments: dict {store_id: {'cd': nome, 'distance_km': dist}}
    store_volumes: dict {store_id: unidades_totais}
    """
    network_cost = 0
    details = {}
    
    for store, data in assignments.items():
        if not data: continue
        
        dist = data['distance_km']
        vol = store_volumes.get(store, 0)
        
        # Proxy simples: 1 viagem por semana? Ou por ciclo de ressuprimento?
        # Vamos assumir que o custo é por ciclo, ou anual se vol for anual.
        
        cost = calculate_transport_cost(dist, vol)
        network_cost += cost
        details[store] = cost
        
    return network_cost, details
