import sys
import os
from pathlib import Path

# Adicionar src/ ao path para importações
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_model_exists():
    """Verifica se o modelo treinado existe"""
    return os.path.exists("models/best_model.pkl")

def train_model():
    """Treina os modelos de IA seguindo o enunciado"""
    print("INICIANDO TREINAMENTO...")
    print("="*60)
    print("Especificações:")
    print("   • Dataset: 250 amostras por classe")
    print("   • Divisão: 80% treino, 10% validação, 10% teste")
    print("   • Algoritmos: k-NN, MLP, Árvore Decisão, Random Forest") 
    print("   • Métricas: Acurácia, Precision, Recall, F-measure")
    print("="*60)
    
    try:
        from data_and_train import train_and_compare_algorithms
        best_name, acc, prec, rec, f1 = train_and_compare_algorithms()
        
        print("\nTREINAMENTO CONCLUÍDO [V]")
        print(f"Melhor algoritmo: {best_name}")
        print(f"Métricas finais:")
        print(f"   • Acurácia: {acc:.3f}")
        print(f"   • Precision: {prec:.3f}")
        print(f"   • Recall: {rec:.3f}")
        print(f"   • F1-Score: {f1:.3f}")
        
        return True
    except Exception as e:
        print(f"\nErro durante o treinamento: {e}")
        return False

def run_game():
    """Executa o front-end"""
    try:
        from game import main as game_main
        game_main()
    except Exception as e:
        print(f"Erro ao executar o jogo: {e}")
        return False
    return True

def run_gui():
    """Executa a interface gráfica (Tkinter)"""
    try:
        from gui import run_gui as _run
        _run()
    except Exception as e:
        print(f"Erro ao executar a GUI: {e}")
        return False
    return True

def show_help():
    """Mostra ajuda do programa"""
    print("TRABALHO DE IA - JOGO DA VELHA")
    print("Opções:")
    print("  --train    Força o re-treinamento da IA")
    print("  --gui      Abre a interface gráfica (Tkinter)")
    print("  --help     Mostra esta mensagem")

def main():
    """Função principal seguindo o enunciado"""
    print("TRABALHO DE IA - JOGO DA VELHA")
    
    # Verifica argumentos
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        return
    
    force_train = "--train" in sys.argv
    use_gui = "--gui" in sys.argv
    
    # Verifica se precisa treinar
    if force_train or not check_model_exists():
        if not check_model_exists():
            print("Modelo não encontrado. Iniciando treinamento...")
        
        if not train_model():
            print("Falha no treinamento. Encerrando...")
            return
    else:
        print("Modelo encontrado [V]")
    
    if use_gui:
        run_gui()
    else:
        run_game()

if __name__ == "__main__":
    main()