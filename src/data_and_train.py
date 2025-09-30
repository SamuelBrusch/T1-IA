# data_and_train.py
# Implementação refatorada com algoritmos separados

import os
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from config import BEST_MODEL_PATH, SCALER_PATH, METADATA_PATH, DATASET_PATH, RESULTS_PATH, MODELS_DIR
from algorithms import KNNAlgorithm, MLPAlgorithm, DecisionTreeAlgorithm, RandomForestAlgorithm

# Configuração
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
COLS = ["tl","tm","tr","ml","mm","mr","bl","bm","br","orig_class"]

# Funções utilitárias
def winner_on_board(board):
    """Verifica se há um vencedor no tabuleiro"""
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if len(board) > max(a,b,c) and board[a] == board[b] == board[c] != 'b':
            return board[a]
    return None

def is_board_full(board):
    """Verifica se o tabuleiro está cheio"""
    return 'b' not in board

def board_state(board):
    """Determina o estado atual do jogo"""
    winner = winner_on_board(board)
    
    if winner == 'x':
        return "X vence"
    elif winner == 'o':
        return "O vence"
    elif is_board_full(board):
        return "Empate"
    
    # Verifica possibilidade de fim de jogo
    for i, cell in enumerate(board):
        if cell == 'b':
            # Testa X
            temp_board = board.copy()
            temp_board[i] = 'x'
            if winner_on_board(temp_board) == 'x':
                return "Possibilidade de Fim de Jogo"
            
            # Testa O
            temp_board[i] = 'o'
            if winner_on_board(temp_board) == 'o':
                return "Possibilidade de Fim de Jogo"
    
    return "Tem jogo"

def generate_all_valid_states():
    """Gera todos os estados válidos do jogo da velha"""
    def is_valid_state(board):
        x_count = board.count('x')
        o_count = board.count('o')
        
        # X sempre joga primeiro
        if not (x_count == o_count or x_count == o_count + 1):
            return False
        
        # Se já há vencedor, não pode ter mais jogadas
        winner = winner_on_board(board)
        if winner:
            # Quem ganhou deve ter feito a última jogada
            if winner == 'x' and x_count != o_count + 1:
                return False
            if winner == 'o' and x_count != o_count:
                return False
        
        return True
    
    def generate_states(board, turn):
        if is_valid_state(board):
            yield board.copy()
        
        if winner_on_board(board) or is_board_full(board):
            return
        
        for i in range(9):
            if board[i] == 'b':
                board[i] = turn
                yield from generate_states(board, 'o' if turn == 'x' else 'x')
                board[i] = 'b'
    
    all_states = []
    initial_board = ['b'] * 9
    
    for state in generate_states(initial_board, 'x'):
        all_states.append(state)
    
    return all_states

def create_balanced_dataset():
    """Cria dataset balanceado com 250 amostras por classe (quando possível)"""
    print("=== CRIANDO DATASET BALANCEADO ===")
    print("Gerando todos os estados válidos...")
    
    all_states = generate_all_valid_states()
    print(f"Estados válidos gerados: {len(all_states)}")
    
    # Classifica todos os estados
    data_by_class = {}
    for board in all_states:
        label = board_state(board)
        if label not in data_by_class:
            data_by_class[label] = []
        data_by_class[label].append(board)
    
    print("Estados por classe:")
    for label, states in data_by_class.items():
        print(f"  {label}: {len(states)}")
    
    # Balanceia o dataset
    target_per_class = 250
    balanced_data = []
    
    for label, states in data_by_class.items():
        if len(states) >= target_per_class:
            selected = random.sample(states, target_per_class)
        else:
            selected = states
            print(f"Aviso: Classe '{label}' tem apenas {len(states)} amostras")
        
        for board in selected:
            row = board + [label]
            balanced_data.append(row)
    
    cols = ["tl","tm","tr","ml","mm","mr","bl","bm","br","label"]
    df = pd.DataFrame(balanced_data, columns=cols)
    
    print(f"Dataset final: {len(df)} amostras")
    print("Distribuição:")
    print(df["label"].value_counts())
    
    return df

def split_dataset(df):
    """Divisão FÍSICA do dataset: 80% treino, 10% validação, 10% teste"""
    print("\n=== DIVISÃO FÍSICA DO DATASET ===")
    print("80% treino, 10% validação, 10% teste")
    
    # Conversão correta dos dados
    X = df[COLS[:-1]].copy()
    for col in X.columns:
        X[col] = X[col].map({'x':2,'o':1,'b':0})
    
    y = df["label"]
    
    # Primeira divisão: 80% treino, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    
    # Segunda divisão: 10% validação, 10% teste
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    
    print(f"Treino: {len(X_train)} amostras")
    print(f"Validação: {len(X_val)} amostras")
    print(f"Teste: {len(X_test)} amostras")
    
    print("\nDistribuição por classe:")
    train_counts = y_train.value_counts()
    val_counts = y_val.value_counts()
    test_counts = y_test.value_counts()
    
    for label in y.unique():
        train_c = train_counts.get(label, 0)
        val_c = val_counts.get(label, 0)
        test_c = test_counts.get(label, 0)
        print(f"  {label}: Treino={train_c} | Val={val_c} | Teste={test_c}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_compare_algorithms():
    """Treina e compara os 4 algoritmos usando módulos organizados"""
    print("\n=== TREINAMENTO AVANÇADO DOS ALGORITMOS ===")
    
    # Cria dataset
    df = create_balanced_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    
    # Garantir dados numéricos
    X_train = X_train.astype(int)
    X_val = X_val.astype(int) 
    X_test = X_test.astype(int)
    
    # Padronização para algoritmos que precisam
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float64))
    X_val_scaled = scaler.transform(X_val.values.astype(np.float64))
    X_test_scaled = scaler.transform(X_test.values.astype(np.float64))
    
    # Inicializa algoritmos com chaves explícitas para salvar modelos
    algorithms = [
        ("knn", KNNAlgorithm(random_state=SEED)),
        ("tree", DecisionTreeAlgorithm(random_state=SEED)),
        ("mlp", MLPAlgorithm(random_state=SEED)),
        ("forest", RandomForestAlgorithm(random_state=SEED))
    ]
    
    # Treinamento dos algoritmos
    all_models = {}
    all_results = {}
    best_model = None
    best_score = -1
    best_name = ""
    best_algorithm = None
    
    for key, algorithm in algorithms:
        print(f"\\n=== Treinando {algorithm.name} ===")
        
        # Treina e avalia o algoritmo
        algo_info = algorithm.train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test, scaler
        )
        
        # Armazena informações usando a chave explícita
        all_models[key] = algo_info
        # Guarda resultados individuais (descrições das configs)
        for cfg_desc, scores in algo_info['results'].items():
            all_results[f"{key}:{cfg_desc}"] = scores
        
        # Verifica se é o melhor modelo
        if algo_info['score'] > best_score:
            best_score = algo_info['score']
            best_model = algo_info['model']
            best_name = algo_info['name']
            best_algorithm = algorithm
    
    print(f"\\n=== MELHOR MODELO: {best_name} (Score: {best_score:.3f}) ===")
    
    # Salva todos os modelos individuais
    print("\\n=== SALVANDO MODELOS INDIVIDUAIS ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for algo_name, model_info in all_models.items():
        if model_info['model'] is not None:
            model_path = os.path.join(MODELS_DIR, f"{algo_name}_model.pkl")
            joblib.dump(model_info['model'], model_path)
            print(f"[V] {model_info['name']} salvo em {model_path}")
            
            # Salva scaler se necessário
            if model_info['scaler'] is not None:
                scaler_path = os.path.join(MODELS_DIR, f"{algo_name}_scaler.pkl")
                joblib.dump(model_info['scaler'], scaler_path)
                print(f"[V] Scaler do {algo_name} salvo em {scaler_path}")
    
    # Salva metadata dos modelos
    metadata = {
        'models': {name: {'name': info['name'], 'score': info['score']} 
                  for name, info in all_models.items() if info['model'] is not None},
        'best_model': best_name,
        'best_score': best_score
    }
    metadata_path = os.path.join(MODELS_DIR, "models_metadata.pkl")
    joblib.dump(metadata, METADATA_PATH)
    print(f"[V] Metadata salva em {METADATA_PATH}")
    
    # Avalia no conjunto de teste
    print("\\n=== AVALIAÇÃO NO CONJUNTO DE TESTE ===")
    
    # Faz predições com o melhor modelo
    if best_algorithm.needs_scaling:
        y_pred = best_model.predict(X_test_scaled)
        joblib.dump(best_model, BEST_MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    else:
        y_pred = best_model.predict(X_test)
        joblib.dump(best_model, BEST_MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            os.remove(SCALER_PATH)
    
    # Métricas detalhadas
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    print(f"Melhor modelo: {best_name}")
    print(f"Acurácia: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    print(f"\\nRelatório detalhado:")
    print(classification_report(y_test, y_pred))
    
    # Salva resultados
    results_data = []
    for algo_name, results in all_results.items():
        if isinstance(results, dict) and 'val' in results:
            results_data.append([algo_name, results['val']])
        else:
            results_data.append([algo_name, results])
    
    results_df = pd.DataFrame(results_data, columns=['Algorithm', 'Validation_Score'])
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\\nResultados salvos em '{RESULTS_PATH}'")
    
    return best_name, acc, prec, rec, f1