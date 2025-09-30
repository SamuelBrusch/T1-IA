"""
Algoritmo MLP (Multi-Layer Perceptron) para o jogo da velha.
"""

from typing import Any, Dict, List
from sklearn.neural_network import MLPClassifier
from .base_algorithm import BaseAlgorithm


class MLPAlgorithm(BaseAlgorithm):
    """Implementação do algoritmo MLP com configurações sofisticadas"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("MLP (Rede Neural)", random_state)
        self.needs_scaling = True  # MLP precisa de escalonamento
    
    def get_configurations(self) -> List[Dict[str, Any]]:
        """Retorna configurações sofisticadas para MLP"""
        configs = []
        
        # Configurações com solver lbfgs (para redes pequenas)
        lbfgs_configs = [
            {'hidden_layer_sizes': (50,), 'solver': 'lbfgs', 'alpha': 0.001, 'max_iter': 1000},
            {'hidden_layer_sizes': (100,), 'solver': 'lbfgs', 'alpha': 0.01, 'max_iter': 1000},
            {'hidden_layer_sizes': (150,), 'solver': 'lbfgs', 'alpha': 0.1, 'max_iter': 1000},
        ]
        configs.extend(lbfgs_configs)
        
        # Configurações com solver adam (para redes maiores)
        adam_configs = [
            {'hidden_layer_sizes': (50, 50), 'solver': 'adam', 'alpha': 0.001, 'learning_rate': 'adaptive', 'max_iter': 2000},
            {'hidden_layer_sizes': (100, 50), 'solver': 'adam', 'alpha': 0.01, 'learning_rate': 'constant', 'max_iter': 2000},
            {'hidden_layer_sizes': (100, 100), 'solver': 'adam', 'alpha': 0.001, 'learning_rate': 'invscaling', 'max_iter': 2000},
            {'hidden_layer_sizes': (150, 75), 'solver': 'adam', 'alpha': 0.01, 'learning_rate': 'adaptive', 'max_iter': 2000},
            {'hidden_layer_sizes': (200, 100, 50), 'solver': 'adam', 'alpha': 0.001, 'learning_rate': 'constant', 'max_iter': 3000},
        ]
        configs.extend(adam_configs)
        
        # Configurações com solver sgd
        sgd_configs = [
            {'hidden_layer_sizes': (100,), 'solver': 'sgd', 'alpha': 0.01, 'learning_rate': 'adaptive', 'momentum': 0.9, 'max_iter': 2000},
            {'hidden_layer_sizes': (100, 50), 'solver': 'sgd', 'alpha': 0.001, 'learning_rate': 'invscaling', 'momentum': 0.95, 'max_iter': 2000},
        ]
        configs.extend(sgd_configs)
        
        return configs
    
    def create_model(self, config: Dict[str, Any]) -> MLPClassifier:
        """Cria modelo MLP com configuração especificada"""
        # Adiciona configurações padrão
        default_config = {
            'random_state': self.random_state,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
        
        # Combina configurações
        final_config = {**default_config, **config}
        
        return MLPClassifier(**final_config)
    
    def get_config_description(self, config: Dict[str, Any]) -> str:
        """Retorna descrição legível da configuração"""
        topology = str(config['hidden_layer_sizes'])
        solver = config['solver']
        alpha = config.get('alpha', 'default')
        return f"{topology}_{solver}_alpha{alpha}"