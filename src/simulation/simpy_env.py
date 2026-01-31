"""
Ambiente de Simulação com SimPy para Gestão de Estoques
Suporta lead time estocástico e múltiplas políticas de reposição.
"""

import simpy
import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Configuração da simulação."""
    horizon_days: int = 365
    warmup_days: int = 30
    n_replications: int = 30
    random_seed: int = 42
    
    # Parâmetros de demanda
    demand_mean: float = 10.0
    demand_std: float = 3.0
    demand_distribution: str = "normal"  # "normal", "poisson", "empirical"
    
    # Parâmetros de lead time
    lead_time_mean: float = 5.0
    lead_time_std: float = 1.5
    lead_time_min: float = 2.0
    lead_time_max: float = 10.0
    lead_time_distribution: str = "truncnorm"  # "constant", "truncnorm", "triangular"
    
    # Custos
    ordering_cost: float = 100.0  # K
    holding_cost_rate: float = 0.20  # h como % do valor
    unit_cost: float = 10.0  # Custo unitário
    stockout_cost: float = 50.0  # Custo por unidade de falta
    
    # Nível de serviço alvo
    csl_target: float = 0.95


@dataclass
class InventoryMetrics:
    """Métricas coletadas durante a simulação."""
    total_demand: float = 0.0
    total_fulfilled: float = 0.0
    total_stockouts: float = 0.0
    total_orders: int = 0
    total_order_quantity: float = 0.0
    avg_inventory: float = 0.0
    max_inventory: float = 0.0
    total_holding_cost: float = 0.0
    total_ordering_cost: float = 0.0
    total_stockout_cost: float = 0.0
    days_with_stockout: int = 0
    
    inventory_history: List[float] = field(default_factory=list)
    demand_history: List[float] = field(default_factory=list)
    
    @property
    def total_cost(self) -> float:
        return self.total_holding_cost + self.total_ordering_cost + self.total_stockout_cost
    
    @property
    def fill_rate(self) -> float:
        if self.total_demand > 0:
            return self.total_fulfilled / self.total_demand
        return 1.0
    
    @property
    def csl(self) -> float:
        """Cycle Service Level aproximado."""
        if self.total_orders > 0:
            cycles_without_stockout = self.total_orders - self.days_with_stockout
            return max(0, cycles_without_stockout / self.total_orders)
        return 1.0


class DemandGenerator:
    """Gerador de demanda para a simulação."""
    
    def __init__(self, config: SimulationConfig, empirical_data: np.ndarray = None):
        self.config = config
        self.empirical_data = empirical_data
        
    def generate(self, rng: np.random.Generator) -> float:
        if self.config.demand_distribution == "normal":
            demand = rng.normal(self.config.demand_mean, self.config.demand_std)
        elif self.config.demand_distribution == "poisson":
            demand = rng.poisson(self.config.demand_mean)
        elif self.config.demand_distribution == "empirical" and self.empirical_data is not None:
            demand = rng.choice(self.empirical_data)
        else:
            demand = rng.normal(self.config.demand_mean, self.config.demand_std)
        
        return max(0, demand)


class LeadTimeGenerator:
    """Gerador de lead time estocástico."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
    def generate(self, rng: np.random.Generator) -> float:
        if self.config.lead_time_distribution == "constant":
            return self.config.lead_time_mean
        
        elif self.config.lead_time_distribution == "truncnorm":
            # Distribuição normal truncada
            a = (self.config.lead_time_min - self.config.lead_time_mean) / self.config.lead_time_std
            b = (self.config.lead_time_max - self.config.lead_time_mean) / self.config.lead_time_std
            lt = truncnorm.rvs(a, b, 
                               loc=self.config.lead_time_mean, 
                               scale=self.config.lead_time_std,
                               random_state=rng)
            return max(1, lt)
        
        elif self.config.lead_time_distribution == "triangular":
            lt = rng.triangular(
                self.config.lead_time_min,
                self.config.lead_time_mean,
                self.config.lead_time_max
            )
            return max(1, lt)
        
        return self.config.lead_time_mean


class InventoryPolicy:
    """Classe base para políticas de estoque."""
    
    def __init__(self, name: str):
        self.name = name
    
    def should_order(self, inventory_position: float, **kwargs) -> bool:
        raise NotImplementedError
    
    def order_quantity(self, inventory_position: float, **kwargs) -> float:
        raise NotImplementedError


class RQPolicy(InventoryPolicy):
    """
    Política (R, Q) - Continuous Review
    Quando IP <= R, pedir Q unidades.
    """
    
    def __init__(self, R: float, Q: float):
        super().__init__(f"(R,Q): R={R:.1f}, Q={Q:.1f}")
        self.R = R
        self.Q = Q
    
    def should_order(self, inventory_position: float, **kwargs) -> bool:
        return inventory_position <= self.R
    
    def order_quantity(self, inventory_position: float, **kwargs) -> float:
        return self.Q


class sSPolicy(InventoryPolicy):
    """
    Política (s, S) - Min-Max
    Quando IP <= s, pedir até S.
    """
    
    def __init__(self, s: float, S: float):
        super().__init__(f"(s,S): s={s:.1f}, S={S:.1f}")
        self.s = s
        self.S = S
    
    def should_order(self, inventory_position: float, **kwargs) -> bool:
        return inventory_position <= self.s
    
    def order_quantity(self, inventory_position: float, **kwargs) -> float:
        return self.S - inventory_position


class PSPolicy(InventoryPolicy):
    """
    Política (P, S) - Periodic Review
    A cada P períodos, pedir até S.
    """
    
    def __init__(self, P: int, S: float):
        super().__init__(f"(P,S): P={P} dias, S={S:.1f}")
        self.P = P
        self.S = S
        self.last_review = -self.P  # Para revisar no dia 0
    
    def should_order(self, inventory_position: float, current_day: int = 0, **kwargs) -> bool:
        if current_day - self.last_review >= self.P:
            self.last_review = current_day
            return True
        return False
    
    def order_quantity(self, inventory_position: float, **kwargs) -> float:
        return max(0, self.S - inventory_position)


class InventorySimulation:
    """
    Simulação de estoque com SimPy.
    """
    
    def __init__(self, config: SimulationConfig, policy: InventoryPolicy, 
                 empirical_demand: np.ndarray = None):
        self.config = config
        self.policy = policy
        self.demand_gen = DemandGenerator(config, empirical_demand)
        self.lead_time_gen = LeadTimeGenerator(config)
        
        # Estado
        self.env = None
        self.inventory_on_hand = 0.0
        self.inventory_on_order = 0.0
        self.metrics = None
        self.rng = None
        
    @property
    def inventory_position(self) -> float:
        return self.inventory_on_hand + self.inventory_on_order
    
    def run(self, seed: int = None) -> InventoryMetrics:
        """Executa uma replicação da simulação."""
        self.env = simpy.Environment()
        self.metrics = InventoryMetrics()
        self.rng = np.random.default_rng(seed or self.config.random_seed)
        
        # Estado inicial
        initial_inv = self.config.demand_mean * self.config.lead_time_mean * 2
        self.inventory_on_hand = initial_inv
        self.inventory_on_order = 0.0
        
        # Processos
        self.env.process(self._daily_process())
        
        # Executar
        self.env.run(until=self.config.warmup_days + self.config.horizon_days)
        
        # Calcular métricas finais
        if self.metrics.inventory_history:
            self.metrics.avg_inventory = np.mean(self.metrics.inventory_history)
            self.metrics.max_inventory = np.max(self.metrics.inventory_history)
        
        return self.metrics
    
    def _daily_process(self):
        """Processo diário: demanda → revisão → custos."""
        day = 0
        
        while True:
            # Fase 1: Chegada de demanda
            demand = self.demand_gen.generate(self.rng)
            
            if day >= self.config.warmup_days:
                self.metrics.demand_history.append(demand)
                self.metrics.total_demand += demand
            
            # Fase 2: Atender demanda
            fulfilled = min(demand, self.inventory_on_hand)
            stockout = demand - fulfilled
            self.inventory_on_hand -= fulfilled
            
            if day >= self.config.warmup_days:
                self.metrics.total_fulfilled += fulfilled
                self.metrics.total_stockouts += stockout
                if stockout > 0:
                    self.metrics.days_with_stockout += 1
                    self.metrics.total_stockout_cost += stockout * self.config.stockout_cost
            
            # Fase 3: Verificar política de reposição
            if self.policy.should_order(self.inventory_position, current_day=day):
                order_qty = self.policy.order_quantity(self.inventory_position)
                if order_qty > 0:
                    self.inventory_on_order += order_qty
                    self.env.process(self._order_arrival(order_qty))
                    
                    if day >= self.config.warmup_days:
                        self.metrics.total_orders += 1
                        self.metrics.total_order_quantity += order_qty
                        self.metrics.total_ordering_cost += self.config.ordering_cost
            
            # Fase 4: Custos de holding
            if day >= self.config.warmup_days:
                h_daily = (self.config.holding_cost_rate * self.config.unit_cost) / 365
                self.metrics.total_holding_cost += self.inventory_on_hand * h_daily
                self.metrics.inventory_history.append(self.inventory_on_hand)
            
            day += 1
            yield self.env.timeout(1)
    
    def _order_arrival(self, quantity: float):
        """Processo de chegada de pedido após lead time."""
        lead_time = self.lead_time_gen.generate(self.rng)
        yield self.env.timeout(lead_time)
        
        self.inventory_on_hand += quantity
        self.inventory_on_order -= quantity


def run_monte_carlo(config: SimulationConfig, policy: InventoryPolicy,
                    empirical_demand: np.ndarray = None,
                    n_replications: int = None) -> Dict:
    """
    Executa simulação Monte Carlo com múltiplas replicações.
    Retorna estatísticas agregadas.
    """
    n_reps = n_replications or config.n_replications
    results = []
    
    for i in range(n_reps):
        sim = InventorySimulation(config, policy, empirical_demand)
        metrics = sim.run(seed=config.random_seed + i)
        results.append({
            'replication': i,
            'total_cost': metrics.total_cost,
            'holding_cost': metrics.total_holding_cost,
            'ordering_cost': metrics.total_ordering_cost,
            'stockout_cost': metrics.total_stockout_cost,
            'fill_rate': metrics.fill_rate,
            'avg_inventory': metrics.avg_inventory,
            'total_stockouts': metrics.total_stockouts,
            'total_orders': metrics.total_orders
        })
    
    df = pd.DataFrame(results)
    
    # Estatísticas
    summary = {}
    for col in ['total_cost', 'holding_cost', 'ordering_cost', 'stockout_cost', 
                'fill_rate', 'avg_inventory', 'total_stockouts']:
        summary[f'{col}_mean'] = df[col].mean()
        summary[f'{col}_std'] = df[col].std()
        summary[f'{col}_ci_lower'] = df[col].mean() - 1.96 * df[col].std() / np.sqrt(n_reps)
        summary[f'{col}_ci_upper'] = df[col].mean() + 1.96 * df[col].std() / np.sqrt(n_reps)
    
    summary['policy_name'] = policy.name
    summary['n_replications'] = n_reps
    summary['all_results'] = df
    
    return summary


def compare_policies(config: SimulationConfig, policies: List[InventoryPolicy],
                     empirical_demand: np.ndarray = None) -> pd.DataFrame:
    """
    Compara múltiplas políticas de estoque via simulação.
    """
    comparison = []
    
    for policy in policies:
        result = run_monte_carlo(config, policy, empirical_demand)
        comparison.append({
            'Política': policy.name,
            'Custo Total (média)': result['total_cost_mean'],
            'Custo Total (IC 95%)': f"[{result['total_cost_ci_lower']:.2f}, {result['total_cost_ci_upper']:.2f}]",
            'Fill Rate (%)': result['fill_rate_mean'] * 100,
            'Estoque Médio': result['avg_inventory_mean'],
            'Faltas Totais': result['total_stockouts_mean'],
            'Custo Holding': result['holding_cost_mean'],
            'Custo Pedidos': result['ordering_cost_mean'],
            'Custo Falta': result['stockout_cost_mean']
        })
    
    return pd.DataFrame(comparison)


def create_policies_from_params(demand_mean: float, demand_std: float,
                                 lead_time_mean: float, lead_time_std: float,
                                 csl_target: float, K: float, h: float) -> List[InventoryPolicy]:
    """
    Cria conjunto de políticas para comparação baseado nos parâmetros.
    """
    from scipy.stats import norm
    
    # Cálculos base
    z = norm.ppf(csl_target)
    D_annual = demand_mean * 365
    
    # EOQ
    Q_eoq = np.sqrt(2 * K * D_annual / h)
    
    # Demanda durante lead time
    mu_L = demand_mean * lead_time_mean
    sigma_L = np.sqrt(demand_std**2 * lead_time_mean + demand_mean**2 * lead_time_std**2)
    
    # Estoque de segurança
    SS = z * sigma_L
    R = mu_L + SS
    
    # (s, S) - s é como R, S é baseado em EOQ
    s = R
    S = s + Q_eoq
    
    # (P, S) - revisão periódica
    P = max(1, int(Q_eoq / demand_mean))  # Período de revisão
    mu_P_L = demand_mean * (P + lead_time_mean)
    sigma_P_L = np.sqrt(demand_std**2 * (P + lead_time_mean) + demand_mean**2 * lead_time_std**2)
    SS_periodic = z * sigma_P_L
    S_periodic = mu_P_L + SS_periodic
    
    policies = [
        RQPolicy(R=R, Q=Q_eoq),
        sSPolicy(s=s, S=S),
        PSPolicy(P=P, S=S_periodic),
        # Variações
        RQPolicy(R=R * 1.1, Q=Q_eoq),  # R 10% maior
        RQPolicy(R=R * 0.9, Q=Q_eoq),  # R 10% menor
    ]
    
    return policies
