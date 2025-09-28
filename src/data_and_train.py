# data_and_train.py
# Implementação corrigida seguindo EXATAMENTE o enunciado
# Todas as variáveis terminam com _$ (simulado como _s para compatibilidade Python)

import os
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from config import BEST_MODEL_PATH, SCALER_PATH, METADATA_PATH, DATASET_PATH, RESULTS_PATH

# Configuração com variáveis personalizadas (terminando com _s para simular _$)
SEED_s = 42
random.seed(SEED_s)
np.random.seed(SEED_s)

UCI_URL_s = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
COLS_s = ["tl_s","tm_s","tr_s","ml_s","mm_s","mr_s","bl_s","bm_s","br_s","orig_class_s"]

# Funções utilitárias personalizadas
def winner_on_board_s(board_s):
    """Verifica se há um vencedor no tabuleiro"""
    wins_s = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a_s,b_s,c_s in wins_s:
        if board_s[a_s] != 'b' and board_s[a_s]==board_s[b_s]==board_s[c_s]:
            return board_s[a_s]
    return None

def is_board_full_s(board_s):
    """Verifica se o tabuleiro está cheio"""
    return all(cell_s != 'b' for cell_s in board_s)

def board_state_s(board_s):
    """Classifica o estado atual do tabuleiro - MÉTODO PARA CONTROLE DO JOGO"""
    winner_s = winner_on_board_s(board_s)
    if winner_s == 'x': 
        return "X vence"
    if winner_s == 'o': 
        return "O vence"
    if is_board_full_s(board_s):
        return "Empate"
    
    # Verifica possibilidade de fim imediato
    for player_s in ['x','o']:
        for i_s in range(9):
            if board_s[i_s] == 'b':
                temp_s = board_s.copy()
                temp_s[i_s] = player_s
                if winner_on_board_s(temp_s) == player_s:
                    return "Possibilidade de Fim de Jogo"
    
    return "Tem jogo"

def generate_all_states_s():
    """Gera todos os estados válidos do jogo"""
    import itertools
    
    print("Gerando todos os estados válidos...")
    positions_s = ['x', 'o', 'b']
    all_data_s = []
    
    for board_s in itertools.product(positions_s, repeat=9):
        board_list_s = list(board_s)
        x_count_s = board_list_s.count('x')
        o_count_s = board_list_s.count('o')
        
        # Regras do jogo da velha
        if not (x_count_s >= o_count_s and x_count_s - o_count_s <= 1):
            continue
            
        # Valida se jogo parou no momento certo
        winner_s = winner_on_board_s(board_list_s)
        if winner_s:
            if winner_s == 'x' and x_count_s != o_count_s + 1:
                continue
            if winner_s == 'o' and x_count_s != o_count_s:
                continue
        
        label_s = board_state_s(board_list_s)
        all_data_s.append({
            'board_s': board_list_s.copy(),
            'label_s': label_s
        })
    
    print(f"Estados válidos gerados: {len(all_data_s)}")
    return all_data_s

def create_balanced_dataset_s():
    """Cria dataset balanceado com EXATAMENTE 250 amostras por classe"""
    print("=== CRIANDO DATASET BALANCEADO ===")
    
    # Gera todos os estados
    all_states_s = generate_all_states_s()
    
    # Separa por classe
    classes_s = {}
    for state_s in all_states_s:
        label_s = state_s['label_s']
        if label_s not in classes_s:
            classes_s[label_s] = []
        classes_s[label_s].append(state_s)
    
    print("Estados por classe:")
    for label_s, states_s in classes_s.items():
        print(f"  {label_s}: {len(states_s)}")
    
    # Balanceia para EXATAMENTE 250 por classe
    target_per_class_s = 250
    balanced_data_s = []
    
    for label_s, states_s in classes_s.items():
        if len(states_s) >= target_per_class_s:
            sampled_s = random.sample(states_s, target_per_class_s)
        else:
            sampled_s = states_s
            print(f"⚠️ Classe '{label_s}' tem apenas {len(states_s)} amostras")
        
        balanced_data_s.extend(sampled_s)
    
    # Converte para DataFrame
    data_rows_s = []
    for item_s in balanced_data_s:
        row_s = item_s['board_s'] + [item_s['label_s']]
        data_rows_s.append(row_s)
    
    cols_s = ["tl_s","tm_s","tr_s","ml_s","mm_s","mr_s","bl_s","bm_s","br_s","label_s"]
    df_s = pd.DataFrame(data_rows_s, columns=cols_s)
    
    print(f"Dataset final: {len(df_s)} amostras")
    print("Distribuição:")
    print(df_s["label_s"].value_counts())
    
    return df_s

def split_dataset_s(df_s):
    """Divisão FÍSICA do dataset: 80% treino, 10% validação, 10% teste"""
    print("\n=== DIVISÃO FÍSICA DO DATASET ===")
    print("80% treino, 10% validação, 10% teste")
    
    # Conversão correta dos dados
    X_s = df_s[COLS_s[:-1]].copy()
    for col in X_s.columns:
        X_s[col] = X_s[col].map({'x':2,'o':1,'b':0})
    
    y_s = df_s["label_s"]
    
    # Primeira divisão: 80% treino, 20% temp
    X_train_s, X_temp_s, y_train_s, y_temp_s = train_test_split(
        X_s, y_s, test_size=0.2, stratify=y_s, random_state=SEED_s
    )
    
    # Segunda divisão: 10% validação, 10% teste
    X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(
        X_temp_s, y_temp_s, test_size=0.5, stratify=y_temp_s, random_state=SEED_s
    )
    
    print(f"Treino: {len(X_train_s)} amostras")
    print(f"Validação: {len(X_val_s)} amostras")
    print(f"Teste: {len(X_test_s)} amostras")
    
    # Verifica distribuição por classe
    print("\nDistribuição por classe:")
    train_counts_s = y_train_s.value_counts()
    val_counts_s = y_val_s.value_counts()
    test_counts_s = y_test_s.value_counts()
    
    for label_s in y_s.unique():
        train_c_s = train_counts_s.get(label_s, 0)
        val_c_s = val_counts_s.get(label_s, 0)
        test_c_s = test_counts_s.get(label_s, 0)
        print(f"  {label_s}: Treino={train_c_s} | Val={val_c_s} | Teste={test_c_s}")
    
    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s

def train_and_compare_algorithms_s():
    """Treina e compara os 4 algoritmos exigidos pelo enunciado"""
    print("\n=== TREINAMENTO DOS ALGORITMOS ===")
    
    # Cria dataset
    df_s = create_balanced_dataset_s()
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = split_dataset_s(df_s)
    
    # Garantir dados numéricos
    X_train_s = X_train_s.astype(int)
    X_val_s = X_val_s.astype(int) 
    X_test_s = X_test_s.astype(int)
    
    # Padronização apenas para MLP
    scaler_s = StandardScaler()
    X_train_scaled_s = scaler_s.fit_transform(X_train_s.values.astype(np.float64))
    X_val_scaled_s = scaler_s.transform(X_val_s.values.astype(np.float64))
    X_test_scaled_s = scaler_s.transform(X_test_s.values.astype(np.float64))
    
    results_s = {}
    best_model_s = None
    best_score_s = -1
    best_name_s = ""
    best_scaler_s = None
    
    print("\\n--- 1. k-NN ---")
    # Testando diferentes valores de k
    k_values_s = [1, 3, 5, 7, 9, 11]
    for k_s in k_values_s:
        model_s = KNeighborsClassifier(n_neighbors=k_s)
        model_s.fit(X_train_s, y_train_s)
        score_s = model_s.score(X_val_s, y_val_s)
        results_s[f"kNN_k{k_s}"] = score_s
        print(f"  k={k_s}: {score_s:.3f}")
        
        if score_s > best_score_s:
            best_score_s, best_model_s, best_name_s = score_s, model_s, f"k-NN (k={k_s})"
    
    print("\\n--- 2. Árvore de Decisão ---")
    # Testando diferentes profundidades
    depths_s = [None, 3, 5, 10, 15, 20]
    for depth_s in depths_s:
        model_s = DecisionTreeClassifier(max_depth=depth_s, random_state=SEED_s)
        model_s.fit(X_train_s, y_train_s)
        score_s = model_s.score(X_val_s, y_val_s)
        results_s[f"DecisionTree_depth{depth_s}"] = score_s
        print(f"  max_depth={depth_s}: {score_s:.3f}")
        
        if score_s > best_score_s:
            best_score_s, best_model_s, best_name_s = score_s, model_s, f"Decision Tree (depth={depth_s})"
    
    print("\\n--- 3. MLP (Rede Neural) ---")
    # Testando diferentes topologias - versão mais robusta
    topologies_s = [(50,), (100,), (150,), (50, 50), (100, 50), (50, 25)]
    for topo_s in topologies_s:
        try:
            model_s = MLPClassifier(
                hidden_layer_sizes=topo_s, 
                max_iter=1000, 
                random_state=SEED_s,
                early_stopping=True,
                validation_fraction=0.1,
                solver='lbfgs' if len(topo_s) == 1 and topo_s[0] <= 100 else 'adam'
            )
            model_s.fit(X_train_scaled_s, y_train_s)
            score_s = model_s.score(X_val_scaled_s, y_val_s)
            results_s[f"MLP_{topo_s}"] = score_s
            print(f"  topologia={topo_s}: {score_s:.3f}")
            
            if score_s > best_score_s:
                best_score_s, best_model_s, best_name_s = score_s, (model_s, scaler_s), f"MLP {topo_s}"
                best_scaler_s = scaler_s
        except Exception as e:
            print(f"  ❌ Erro MLP {topo_s}: {e}")
            continue
    
    print("\\n--- 4. Random Forest ---")
    # Testando diferentes números de árvores
    n_trees_s = [50, 100, 150, 200, 300]
    for n_s in n_trees_s:
        model_s = RandomForestClassifier(
            n_estimators=n_s, 
            random_state=SEED_s,
            max_depth=10,  # Evita overfitting
            min_samples_split=5
        )
        model_s.fit(X_train_s, y_train_s)
        score_s = model_s.score(X_val_s, y_val_s)
        results_s[f"RandomForest_n{n_s}"] = score_s
        print(f"  n_estimators={n_s}: {score_s:.3f}")
        
        if score_s > best_score_s:
            best_score_s, best_model_s, best_name_s = score_s, model_s, f"Random Forest (n={n_s})"
    
    print(f"\\n=== MELHOR MODELO: {best_name_s} (Score: {best_score_s:.3f}) ===")
    
    # Avalia no conjunto de teste
    print("\\n=== AVALIAÇÃO NO CONJUNTO DE TESTE ===")
    if isinstance(best_model_s, tuple):  # MLP com scaler
        model_final_s, scaler_final_s = best_model_s
        y_pred_s = model_final_s.predict(X_test_scaled_s)
        joblib.dump(model_final_s, BEST_MODEL_PATH)
        joblib.dump(scaler_final_s, SCALER_PATH)
    else:
        y_pred_s = best_model_s.predict(X_test_s)
        joblib.dump(best_model_s, BEST_MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            os.remove(SCALER_PATH)
    
    # Métricas detalhadas
    acc_s = accuracy_score(y_test_s, y_pred_s)
    prec_s, rec_s, f1_s, _ = precision_recall_fscore_support(y_test_s, y_pred_s, average='macro')
    
    print(f"Melhor modelo: {best_name_s}")
    print(f"Acurácia: {acc_s:.3f}")
    print(f"Precision: {prec_s:.3f}")
    print(f"Recall: {rec_s:.3f}")
    print(f"F1-Score: {f1_s:.3f}")
    
    print("\\nRelatório detalhado:")
    print(classification_report(y_test_s, y_pred_s))
    
    # Salva resultados para análise
    results_df_s = pd.DataFrame(list(results_s.items()), columns=['Algorithm', 'Validation_Score'])
    results_df_s.to_csv(RESULTS_PATH, index=False)
    print(f"\\nResultados salvos em '{RESULTS_PATH}'")
    
    return best_name_s, acc_s, prec_s, rec_s, f1_s

if __name__ == "__main__":
    train_and_compare_algorithms_s()