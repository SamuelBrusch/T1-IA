import os

# Diretórios do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Arquivos principais
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model_s.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_s.pkl')
METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata_s.pkl')
DATASET_PATH = os.path.join(DATA_DIR, 'tic-tac-toe.data')
RESULTS_PATH = os.path.join(DATA_DIR, 'algorithm_comparison.csv')

# Criar diretórios se não existirem
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)