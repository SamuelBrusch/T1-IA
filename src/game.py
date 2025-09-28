# game.py
# Jogo humano vs humano. A IA só classifica o estado do tabuleiro.

import joblib
import numpy as np
import os
import sys
from config import BEST_MODEL_PATH, SCALER_PATH, METADATA_PATH

# Importar do arquivo atualizado
try:
    from data_and_train import winner_on_board_s, is_board_full_s, board_state_s
except ImportError:
    # Fallback para versão antiga
    from data_and_train import winner_on_board, is_board_full, board_state
    # Renomear funções para usar _s
    winner_on_board_s = winner_on_board
    is_board_full_s = is_board_full
    board_state_s = board_state

print("=== Jogo da Velha com IA Classificadora ===")
#print("🎮 Humano vs Humano (dois jogadores)")
#print("🧠 A IA classifica o estado do tabuleiro a cada jogada")
#print("=" * 50)

if not os.path.exists(BEST_MODEL_PATH):
    print(f"\n❌ ERRO: Modelo não encontrado em {BEST_MODEL_PATH}!")
    print("Execute primeiro: python main.py --train")
#    print("Isso irá treinar a IA e criar o modelo necessário.")
    sys.exit(1)

#print("✅ Carregando modelo treinado...")
model_s = joblib.load(BEST_MODEL_PATH)
scaler_s = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
metadata_s = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else None

if metadata_s:
    algorithm_name_s = metadata_s.get("model_name", "Desconhecido")
    print(f"✅ Algoritmo carregado: {algorithm_name_s}")
else:
    print("Modelo carregado!")

class TicTacToeInterface:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler
        self.board = ['b'] * 9
        self.current_player = 'x'
        self.move_history = []
        self.ai_predictions = []
        self.correct_predictions = 0
        self.total_predictions = 0
        
    def show_board(self):
        """Mostra o tabuleiro de forma visual"""
        def display_cell(v): 
            if v == 'x': return '❌'
            elif v == 'o': return '⭕'
            else: return '⬜'
        
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
        real = board_state_s(self.board)  # Usando função com _s
        
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

    def make_move(self, position):
        """Executa uma jogada"""
        if self.board[position] != 'b':
            return False
        
        self.board[position] = self.current_player
        self.move_history.append({
            'position': position,
            'player': self.current_player,
            'move_notation': f"{chr(65 + position % 3)}{position // 3 + 1}"
        })
        
        # Alterna jogador
        self.current_player = 'o' if self.current_player == 'x' else 'x'
        return True

    def get_move_from_player(self):
        """Obtém jogada do jogador atual"""
        player_symbol = '❌' if self.current_player == 'x' else '⭕'
        
        while True:
            try:
                move_str = input(f"\nJogador {player_symbol} ({self.current_player.upper()}), sua vez! Digite a posição (ex: A1, B2, C3): ")
                
                if move_str.lower() in ['quit', 'sair', 'exit']:
                    return None
                
                position = self.parse_move(move_str)
                if position is None:
                    print("❌ Formato inválido! Use formato A1, B2, C3, etc.")
                    continue
                    
                if self.board[position] != 'b':
                    print("❌ Posição já ocupada! Escolha outra.")
                    continue
                    
                return position
                
            except KeyboardInterrupt:
                print("\n\nJogo interrompido pelo usuário.")
                return None
            except:
                print("❌ Entrada inválida. Tente novamente.")

    def show_game_stats(self):
        """Mostra estatísticas da IA durante o jogo"""
        if self.total_predictions > 0:
            accuracy = (self.correct_predictions / self.total_predictions) * 100
            print(f"\n📊 Estatísticas da IA:")
            print(f"   Predições corretas: {self.correct_predictions}/{self.total_predictions}")
            print(f"   Acurácia: {accuracy:.1f}%")

    def show_move_history(self):
        """Mostra histórico de jogadas"""
        if self.move_history:
            print(f"\n📝 Histórico de jogadas:")
            for i, move in enumerate(self.move_history):
                player_symbol = '❌' if move['player'] == 'x' else '⭕'
                print(f"   {i+1}. {player_symbol} {move['player'].upper()} -> {move['move_notation']}")

    def play_game(self):
        """Loop principal do jogo"""
        print(f"\n🎮 Iniciando novo jogo!")
        print("Jogador ❌ (X) começa")
        print("Digite 'sair' a qualquer momento para encerrar")
        
        while True:
            self.show_board()
            
            # IA faz predição antes da jogada
            if not (winner_on_board_s(self.board) or is_board_full_s(self.board)):
                pred, real, correct = self.get_ai_prediction()
                
                status_icon = "✅" if correct else "❌"
                print(f"\n🤖 IA prediz: '{pred}' | Real: '{real}' {status_icon}")
                self.show_game_stats()
            
            # Verifica fim de jogo
            winner = winner_on_board_s(self.board)
            if winner:
                winner_symbol = '❌' if winner == 'x' else '⭕'
                print(f"\n🎉 FIM DE JOGO! Vencedor: {winner_symbol} ({winner.upper()})")
                break
                
            if is_board_full_s(self.board):
                print(f"\n🤝 FIM DE JOGO! EMPATE!")
                break
            
            # Jogada do usuário
            position = self.get_move_from_player()
            if position is None:  # usuário quer sair
                break
                
            self.make_move(position)
        
        # Estatísticas finais
        print(f"\n" + "="*50)
        self.show_move_history()
        self.show_game_stats()
        
        # Detalhes das predições
        if self.ai_predictions:
            print(f"\n🔍 Detalhes das predições da IA:")
            for pred in self.ai_predictions:
                status = "✅" if pred['correct'] else "❌"
                print(f"   Jogada {pred['move']+1}: Predito='{pred['predicted']}' | Real='{pred['real']}' {status}")
        
        print(f"="*50)

def main():
    """Função principal"""
    try:
        interface_s = TicTacToeInterface(model_s, scaler_s)
        
        while True:
            interface_s.play_game()
            
            print(f"\n🎮 Deseja jogar novamente?")
            choice = input("Digite 's' para sim, qualquer outra tecla para sair: ").strip().lower()
            
            if choice not in ['s', 'sim', 'y', 'yes']:
                break
            
            # Reset para novo jogo
            interface_s.board = ['b'] * 9
            interface_s.current_player = 'x'
            interface_s.move_history = []
            interface_s.ai_predictions = []
            interface_s.correct_predictions = 0
            interface_s.total_predictions = 0
        
        print("\n👋 Obrigado por jogar! Até a próxima!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Jogo encerrado pelo usuário. Até a próxima!")

if __name__ == "__main__":
    main()
