"""
Algoritmo Random Forest para o jogo da velha.
"""

from typing import Any, Dict, List
from sklearn.ensemble import RandomForestClassifier
from .base_algorithm import BaseAlgorithm


class RandomForestAlgorithm(BaseAlgorithm):
    """Implementação do algoritmo Random Forest com configurações sofisticadas"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Random Forest", random_state)
        self.needs_scaling = False  # Random Forest não precisa de escalonamento
    
    def get_configurations(self) -> List[Dict[str, Any]]:
        """Retorna configurações sofisticadas para Random Forest"""
        configs = [
            # Configurações básicas com diferentes números de árvores
            {'n_estimators': 50, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 2},
            {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 1},
            {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 2},
            
            # Configurações com max_features diferentes
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
            {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'log2'},
            {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
            
            # Configurações mais conservadoras para evitar overfitting
            {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 5},
            {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 8, 'min_samples_leaf': 3},
            
            # Configurações balanceadas
            {'n_estimators': 120, 'max_depth': 18, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
            {'n_estimators': 250, 'max_depth': 25, 'min_samples_split': 6, 'min_samples_leaf': 3, 'max_features': 'log2'},
        ]
        
        return configs
    
    def create_model(self, config: Dict[str, Any]) -> RandomForestClassifier:
        """Cria modelo Random Forest com configuração especificada"""
        # Adiciona random_state à configuração
        final_config = {**config, 'random_state': self.random_state}
        return RandomForestClassifier(**final_config)
    
    def get_config_description(self, config: Dict[str, Any]) -> str:
        """Retorna descrição legível da configuração"""
        n_est = config['n_estimators']
        depth = config['max_depth'] if config['max_depth'] is not None else 'None'
        max_feat = config.get('max_features', 'all')
        min_split = config['min_samples_split']
        return f"{n_est}trees_depth{depth}_feat{max_feat}_split{min_split}"