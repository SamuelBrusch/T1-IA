"""
Interface do jogo humano vs IA com escolha do algoritmo.
- Humano vs IA: a IA faz jogadas aleatórias (conforme pedido)
- Menu para escolher qual modelo/algoritmo usar (carregado de models/)
"""

import joblib
import numpy as np
import os
import sys
import random
from typing import Optional, Tuple

from config import BEST_MODEL_PATH, SCALER_PATH, METADATA_PATH, MODELS_DIR
from data_and_train import winner_on_board, is_board_full, board_state

ALGORITHM_LABELS = {
    "knn": "k-NN (k Vizinhos)",
    "mlp": "MLP (Rede Neural)",
    "tree": "Árvore de Decisão",
    "forest": "Random Forest",
}

def get_model_paths_for_key(key: str):
    model_path = os.path.join(MODELS_DIR, f"{key}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{key}_scaler.pkl")
    return model_path, scaler_path if os.path.exists(scaler_path) else None

def choose_algorithm() -> Tuple[str, str, Optional[str]]:
    """Menu para o usuário escolher entre os 4 algoritmos fixos."""
    print("\nEscolha o algoritmo contra o qual deseja jogar:")
    keys = ["knn", "mlp", "tree", "forest"]
    for i, key in enumerate(keys, 1):
        label = ALGORITHM_LABELS[key]
        print(f"  {i}. {label}")

    while True:
        choice = input("Escolha [1-4]: ").strip()
        if not choice.isdigit():
            print("Digite um número válido.")
            continue
        idx = int(choice)
        if 1 <= idx <= 4:
            sel = keys[idx-1]
            model_path, scaler_path = get_model_paths_for_key(sel)
            return sel, model_path, scaler_path
        print("Opção inválida.")

def load_selected_model() -> Tuple[object, Optional[object], str]:
    """Carrega o modelo escolhido pelo usuário."""
    algo_key, model_path, scaler_path = choose_algorithm()
    if not os.path.exists(model_path):
        print(f"\nAviso: Modelo '{algo_key}' não encontrado em {model_path}.")
        print("Iniciando treinamento de todos os algoritmos para gerar os modelos...")
        try:
            from data_and_train import train_and_compare_algorithms
            train_and_compare_algorithms()
        except Exception as e:
            print(f"Falha ao treinar modelos: {e}")
            print("Execute manualmente: python main.py --train")
            sys.exit(1)
        # Tenta novamente
        if not os.path.exists(model_path):
            print(f"Após o treinamento, o modelo '{algo_key}' ainda não foi encontrado.")
            print("Tente treinar novamente com: python main.py --train")
            sys.exit(1)
    print(f"Carregando modelo: {os.path.basename(model_path)} [V]")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
    return model, scaler, algo_key

class TicTacToeInterface:
    def __init__(self, model, scaler=None, ai_random_moves=True):
        self.model = model
        self.scaler = scaler
        self.board = ['b'] * 9
        self.current_player = 'x'
        self.move_history = []
        self.ai_predictions = []
        self.correct_predictions = 0
        self.total_predictions = 0
        self.ai_random_moves = ai_random_moves
        
    def show_board(self):
        """Mostra o tabuleiro de forma visual"""
        def display_cell(v): 
            if v == 'x': return 'X'
            elif v == 'o': return 'O'
            else: return '_'
        
        print("\n" + "="*25)
        print("   A   B   C")
        for i in range(3):
            row = f"{i+1}  "
            for j in range(3):
                idx = i*3 + j
                row += display_cell(self.board[idx]) + " "
            print(row)
        print("="*25)
        
        # Mostra legenda de posições
        print("\nPosições (digite A1, B2, C3, etc.):")
        print("   A   B   C")
        for i in range(3):
            row = f"{i+1}  "
            for j in range(3):
                idx = i*3 + j
                if self.board[idx] == 'b':
                    row += f"{chr(65+j)}{i+1} "
                else:
                    row += "   "
            print(row)

    def encode_board(self):
        """Codifica o tabuleiro para o modelo"""
        arr = np.array([[{'x':2,'o':1,'b':0}[v] for v in self.board]])
        if self.scaler: 
            arr = self.scaler.transform(arr)
        return arr

    def get_ai_prediction(self):
        """Obtém predição da IA sobre o estado atual"""
        pred = self.model.predict(self.encode_board())[0]
        real = board_state(self.board)  # Usando função limpa
        
        self.total_predictions += 1
        is_correct = pred == real
        if is_correct:
            self.correct_predictions += 1
        
        self.ai_predictions.append({
            'move': len(self.move_history),
            'predicted': pred,
            'real': real,
            'correct': is_correct
        })
        
        return pred, real, is_correct

    def parse_move(self, move_str):
        """Converte entrada do usuário (A1, B2, etc.) para índice"""
        move_str = move_str.strip().upper()
        if len(move_str) != 2:
            return None
        
        col_char, row_char = move_str[0], move_str[1]
        
        if col_char not in 'ABC' or row_char not in '123':
            return None
            
        col = ord(col_char) - ord('A')  # A=0, B=1, C=2
        row = int(row_char) - 1         # 1=0, 2=1, 3=2
        
        return row * 3 + col

    def get_move_from_player(self):
        """Obtém jogada válida do jogador atual"""
        player_symbol = 'X' if self.current_player == 'x' else 'O'
        
        while True:
            try:
                move_input = input(f"Jogador {player_symbol} ({self.current_player.upper()}), sua vez! Digite a posição (ex: A1, B2, C3): ").strip()
                
                if move_input.lower() == 'sair':
                    print("Jogo encerrado pelo usuário.")
                    sys.exit(0)
                
                position = self.parse_move(move_input)
                
                if position is None:
                    print("Formato inválido! Use formato A1, B2, C3, etc.")
                    continue
                
                if self.board[position] != 'b':
                    print("Posição já ocupada! Escolha outra.")
                    continue
                
                return position
                
            except (KeyboardInterrupt, EOFError):
                print("\nJogo encerrado pelo usuário.")
                sys.exit(0)

    def get_random_ai_move(self):
        """Seleciona uma jogada aleatória válida para a IA"""
        empty_positions = [i for i, v in enumerate(self.board) if v == 'b']
        if not empty_positions:
            return None
        return random.choice(empty_positions)

    def make_move(self, position):
        """Executa uma jogada"""
        if self.board[position] == 'b':
            self.board[position] = self.current_player
            
            move_notation = f"{chr(65 + position % 3)}{position // 3 + 1}"
            self.move_history.append({
                'player': self.current_player,
                'position': position,
                'notation': move_notation
            })
            return True
        return False

    def switch_player(self):
        """Alterna entre jogadores"""
        self.current_player = 'o' if self.current_player == 'x' else 'x'

    def show_game_stats(self):
        """Mostra estatísticas da IA"""
        if self.total_predictions > 0:
            accuracy = (self.correct_predictions / self.total_predictions) * 100
            print(f"\nEstatísticas da IA:")
            print(f"   Predições corretas: {self.correct_predictions}/{self.total_predictions}")
            print(f"   Acurácia: {accuracy:.1f}%")

    def play_game(self):
        """Executa uma partida completa"""
        print("\nIniciando novo jogo!")
        print("Modo: Humano (X) vs IA (O) - IA joga aleatoriamente")
        print("Jogador X começa")
        print("Digite 'sair' a qualquer momento para encerrar")
        
        while True:
            self.show_board()
            
            # IA faz predição antes da jogada
            if not (winner_on_board(self.board) or is_board_full(self.board)):
                pred, real, correct = self.get_ai_prediction()
                
                status_icon = "V" if correct else "X"
                print(f"\nIA prediz: '{pred}' | Real: '{real}' [{status_icon}]")
                self.show_game_stats()
            
            # Verifica fim de jogo
            winner = winner_on_board(self.board)
            if winner:
                winner_symbol = 'X' if winner == 'x' else 'O'
                print(f"\nFIM DE JOGO! Vencedor: {winner_symbol} ({winner.upper()})")
                break
                
            if is_board_full(self.board):
                print(f"\nFIM DE JOGO! EMPATE!")
                break
            
            # Jogada do usuário (humano como 'x')
            if self.current_player == 'x':
                position = self.get_move_from_player()
                self.make_move(position)
                self.switch_player()
                continue

            # Jogada da IA (aleatória), IA joga como 'o'
            if self.current_player == 'o':
                ai_pos = self.get_random_ai_move()
                if ai_pos is not None:
                    self.make_move(ai_pos)
                    print(f"\nIA (O) jogou em: {chr(65 + ai_pos % 3)}{ai_pos // 3 + 1}")
                self.switch_player()
        
        self.show_final_stats()

    def show_final_stats(self):
        """Mostra estatísticas finais da partida"""
        print("\n" + "="*50)
        print("\nHistórico de jogadas:")
        for i, move in enumerate(self.move_history, 1):
            symbol = 'X' if move['player'] == 'x' else 'O'
            print(f"   {i}. {symbol} {move['player'].upper()} -> {move['notation']}")
        
        if self.total_predictions > 0:
            accuracy = (self.correct_predictions / self.total_predictions) * 100
            print(f"\nEstatísticas da IA:")
            print(f"   Predições corretas: {self.correct_predictions}/{self.total_predictions}")
            print(f"   Acurácia: {accuracy:.1f}%")
            
            print(f"\nDetalhes das predições da IA:")
            for i, pred_info in enumerate(self.ai_predictions, 1):
                status = "V" if pred_info['correct'] else "X"
                print(f"   Jogada {i}: Predito='{pred_info['predicted']}' | Real='{pred_info['real']}' [{status}]")
        
        print("=" * 50)

def main():
    """Função principal"""
    try:
        model, scaler, algo_key = load_selected_model()
        print(f"Usando algoritmo: {algo_key}")
        interface = TicTacToeInterface(model, scaler, ai_random_moves=True)
        
        while True:
            interface.play_game()
            
            print(f"\nDeseja jogar novamente?")
            choice = input("Digite 's' para sim, qualquer outra tecla para sair: ").strip().lower()
            
            if choice not in ['s', 'sim', 'y', 'yes']:
                break
            
            # Reset para novo jogo
            interface.board = ['b'] * 9
            interface.current_player = 'x'
            interface.move_history = []
            interface.ai_predictions = []
            interface.correct_predictions = 0
            interface.total_predictions = 0
        
        print("\nObrigado por jogar! Até a próxima!")
        
    except KeyboardInterrupt:
        print("\n\nJogo encerrado pelo usuário. Até a próxima!")

if __name__ == "__main__":
    main()