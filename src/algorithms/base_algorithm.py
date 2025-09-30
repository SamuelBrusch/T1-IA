"""
Classe base para algoritmos de Machine Learning.
Define interface comum para todos os algoritmos.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score


class BaseAlgorithm(ABC):
    """Classe base abstrata para algoritmos de ML"""
    
    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.best_model = None
        self.best_score = -1
        self.best_config = None
        self.scaler = None
        self.needs_scaling = False
    
    @abstractmethod
    def get_configurations(self) -> List[Dict[str, Any]]:
        """Retorna lista de configurações para testar"""
        pass
    
    @abstractmethod
    def create_model(self, config: Dict[str, Any]) -> Any:
        """Cria um modelo com a configuração especificada"""
        pass
    
    @abstractmethod
    def get_config_description(self, config: Dict[str, Any]) -> str:
        """Retorna descrição legível da configuração"""
        pass
    
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          scaler: Optional[Any] = None) -> Dict[str, Any]:
        """
        Treina o algoritmo com todas as configurações e retorna o melhor modelo.
        
        Returns:
            Dict com informações do melhor modelo
        """
        print(f"\\n--- {self.name} ---")
        results = {}
        
        # Prepara dados escalonados se necessário
        if self.needs_scaling and scaler is not None:
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            self.scaler = scaler
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test
        
        # Testa todas as configurações
        for config in self.get_configurations():
            try:
                model = self.create_model(config)
                
                # Treina o modelo
                if self.needs_scaling:
                    model.fit(X_train_scaled, y_train)
                    val_score = model.score(X_val_scaled, y_val)
                    test_score = model.score(X_test_scaled, y_test)
                else:
                    model.fit(X_train, y_train)
                    val_score = model.score(X_val, y_val)
                    test_score = model.score(X_test, y_test)
                
                config_desc = self.get_config_description(config)
                results[config_desc] = {'val': val_score, 'test': test_score}
                print(f"  {config_desc}: Val={val_score:.3f}, Test={test_score:.3f}")
                
                # Atualiza melhor modelo
                if val_score > self.best_score:
                    self.best_score = val_score
                    self.best_model = model
                    self.best_config = config
                    
            except Exception as e:
                config_desc = self.get_config_description(config)
                print(f"  Erro em {config_desc}: {e}")
                continue
        
        return {
            'model': self.best_model,
            'name': f'{self.name} (Melhor: {self.best_score:.3f})',
            'scaler': self.scaler if self.needs_scaling else None,
            'score': self.best_score,
            'config': self.best_config,
            'results': results
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições usando o melhor modelo"""
        if self.best_model is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        if self.needs_scaling and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            return self.best_model.predict(X_scaled)
        else:
            return self.best_model.predict(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo"""
        return {
            'name': self.name,
            'best_score': self.best_score,
            'best_config': self.best_config,
            'needs_scaling': self.needs_scaling
        }