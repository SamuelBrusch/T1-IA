"""
Algoritmo k-NN (k-Nearest Neighbors) para o jogo da velha.
"""

from typing import Any, Dict, List
from sklearn.neighbors import KNeighborsClassifier
from .base_algorithm import BaseAlgorithm


class KNNAlgorithm(BaseAlgorithm):
    """Implementação do algoritmo k-NN com configurações sofisticadas"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("k-NN (k-Nearest Neighbors)", random_state)
        self.needs_scaling = False  # k-NN não precisa de escalonamento para dados categóricos
    
    def get_configurations(self) -> List[Dict[str, Any]]:
        """Retorna configurações sofisticadas para k-NN"""
        configs = []
        
        # Configurações com weights='uniform'
        for k in range(1, 16):  # k de 1 a 15
            configs.append({
                'n_neighbors': k,
                'weights': 'uniform',
                'metric': 'euclidean'
            })
        
        # Configurações com weights='distance'
        for k in range(3, 16):  # k de 3 a 15 (distance weights precisa k > 1)
            configs.append({
                'n_neighbors': k,
                'weights': 'distance',
                'metric': 'euclidean'
            })
        
        # Configurações adicionais com métricas diferentes
        for k in [5, 7, 9, 11]:
            configs.extend([
                {
                    'n_neighbors': k,
                    'weights': 'uniform',
                    'metric': 'manhattan'
                },
                {
                    'n_neighbors': k,
                    'weights': 'distance',
                    'metric': 'manhattan'
                }
            ])
        
        return configs
    
    def create_model(self, config: Dict[str, Any]) -> KNeighborsClassifier:
        """Cria modelo k-NN com configuração especificada"""
        return KNeighborsClassifier(**config)
    
    def get_config_description(self, config: Dict[str, Any]) -> str:
        """Retorna descrição legível da configuração"""
        k = config['n_neighbors']
        weights = config['weights']
        metric = config.get('metric', 'euclidean')
        return f"k={k}, weights={weights}, metric={metric}"