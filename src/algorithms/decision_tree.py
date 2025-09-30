"""
Algoritmo Decision Tree (Árvore de Decisão) para o jogo da velha.
"""

from typing import Any, Dict, List
from sklearn.tree import DecisionTreeClassifier
from .base_algorithm import BaseAlgorithm


class DecisionTreeAlgorithm(BaseAlgorithm):
    """Implementação do algoritmo Decision Tree com configurações sofisticadas"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Árvore de Decisão (Decision Tree)", random_state)
        self.needs_scaling = False  # Decision Tree não precisa de escalonamento
    
    def get_configurations(self) -> List[Dict[str, Any]]:
        """Retorna configurações sofisticadas para Decision Tree"""
        configs = [
            # Configurações com critério gini
            {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
            {'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'gini'},
            {'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
            {'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1, 'criterion': 'gini'},
            {'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'gini'},
            
            # Configurações com critério entropy
            {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy'},
            {'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1, 'criterion': 'entropy'},
            {'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'entropy'},
            
            # Configurações mais restritivas para evitar overfitting
            {'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 5, 'criterion': 'gini'},
            {'max_depth': 12, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'entropy'},
        ]
        
        return configs
    
    def create_model(self, config: Dict[str, Any]) -> DecisionTreeClassifier:
        """Cria modelo Decision Tree com configuração especificada"""
        # Adiciona random_state à configuração
        final_config = {**config, 'random_state': self.random_state}
        return DecisionTreeClassifier(**final_config)
    
    def get_config_description(self, config: Dict[str, Any]) -> str:
        """Retorna descrição legível da configuração"""
        depth = config['max_depth'] if config['max_depth'] is not None else 'None'
        criterion = config['criterion']
        min_split = config['min_samples_split']
        min_leaf = config['min_samples_leaf']
        return f"depth={depth}, criterion={criterion}, split={min_split}, leaf={min_leaf}"