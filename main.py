import sys
import os
from pathlib import Path

# Adicionar src/ ao path para importações
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_model_exists_s():
    """Verifica se o modelo treinado existe"""
    return os.path.exists("models/best_model_s.pkl")

def train_model_s():
    """Treina os modelos de IA seguindo o enunciado"""
    print("🤖 INICIANDO TREINAMENTO...")
    print("="*60)
    print("📋 Especificações:")
    print("   • Dataset: 250 amostras por classe")
    print("   • Divisão: 80% treino, 10% validação, 10% teste")
    print("   • Algoritmos: k-NN, MLP, Árvore Decisão, Random Forest") 
    print("   • Métricas: Acurácia, Precision, Recall, F-measure")
    print("="*60)
    
    try:
        from data_and_train import train_and_compare_algorithms_s
        best_name_s, acc_s, prec_s, rec_s, f1_s = train_and_compare_algorithms_s()
        
        print("\\n✅ TREINAMENTO CONCLUÍDO!")
        print(f"🏆 Melhor algoritmo: {best_name_s}")
        print(f"📊 Métricas finais:")
        print(f"   • Acurácia: {acc_s:.3f}")
        print(f"   • Precision: {prec_s:.3f}")
        print(f"   • Recall: {rec_s:.3f}")
        print(f"   • F1-Score: {f1_s:.3f}")
        
        return True
    except Exception as e:
        print(f"\\n❌ Erro durante o treinamento: {e}")
        return False

def run_game_s():
    """Executa o front-end do jogo (humano vs humano)"""
    try:
        from game import main
        main()
    except Exception as e:
        print(f"❌ Erro ao executar o jogo: {e}")
        return False
    return True

def show_help_s():
    """Mostra ajuda do programa"""
    print(__doc__)
    print("Opções:")
    print("  --train    Força o re-treinamento da IA")
    print("  --help     Mostra esta mensagem")

def main_s():
    """Função principal seguindo o enunciado"""
    print("🎮 TRABALHO DE IA - JOGO DA VELHA")
    
    # Verifica argumentos
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help_s()
        return
    
    force_train_s = "--train" in sys.argv
    
    # Verifica se precisa treinar
    if force_train_s or not check_model_exists_s():
        if not check_model_exists_s():
            print("⚠️  Modelo não encontrado. Iniciando treinamento...")
        
        if not train_model_s():
            print("❌ Falha no treinamento. Encerrando...")
            return
    else:
        print("✅ Modelo encontrado!")
    
    run_game_s()

if __name__ == "__main__":
    main_s()