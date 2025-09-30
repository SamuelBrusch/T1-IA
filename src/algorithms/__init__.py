"""
Módulo de algoritmos de Machine Learning para o jogo da velha.
"""

from .knn import KNNAlgorithm
from .mlp import MLPAlgorithm
from .decision_tree import DecisionTreeAlgorithm
from .random_forest import RandomForestAlgorithm

__all__ = [
    'KNNAlgorithm',
    'MLPAlgorithm', 
    'DecisionTreeAlgorithm',
    'RandomForestAlgorithm'
]