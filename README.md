# 🎮 Jogo da Velha com IA Classificadora

> Trabalho de Inteligência Artificial - Implementação seguindo exatamente o enunciado

## 📋 Especificações Implementadas

- ✅ **4 algoritmos de IA**: k-NN, MLP, Árvore de Decisão, Random Forest
- ✅ **Dataset balanceado**: 250 amostras por classe (exceto Empate)
- ✅ **Divisão física**: 80% treino, 10% validação, 10% teste
- ✅ **Front-end**: Humano vs Humano (dois jogadores)
- ✅ **IA classifica estados**: A cada jogada em tempo real
- ✅ **Variáveis personalizadas**: Todas terminam com `_s` (simulando `_$`)

## 🚀 Como Usar

### 1. Configurar Ambiente

```bash
# Ativar ambiente virtual
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# ou
source .venv/bin/activate     # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar

```bash
# Jogar (usa modelo existente - RÁPIDO)
python main.py

# Treinar novamente (só quando necessário - LENTO)
python main.py --train
```

## 📁 Estrutura do Projeto

```
T1 IA/
├── src/                      # Código fonte
│   ├── config.py            # Configurações e caminhos
│   ├── data_and_train.py    # Treinamento dos algoritmos
│   └── game.py              # Interface do jogo
├── models/                  # Modelos treinados
│   ├── best_model_s.pkl     # Melhor modelo
│   ├── scaler_s.pkl         # Normalizador (se MLP)
│   └── model_metadata_s.pkl # Metadados
├── data/                    # Dados e resultados
│   ├── tic-tac-toe.data     # Dataset UCI
│   └── algorithm_comparison.csv # Resultados dos testes
├── docs/                    # Documentação
│   └── enunciado.pdf        # Enunciado do trabalho
├── main.py                  # Programa principal
├── requirements.txt         # Dependências
├── README.md               # Este arquivo
└── .gitignore              # Arquivos ignorados pelo Git
```

## 🎯 Como Funciona

### Treinamento Inteligente
- **`python main.py`**: Carrega modelo existente (instantâneo)
- **`python main.py --train`**: Força novo treinamento (quando necessário)

### Jogo
1. **Dois jogadores humanos** se alternam
2. **IA classifica** o estado do tabuleiro a cada jogada
3. **Estatísticas em tempo real** da acurácia da IA
4. **Interface intuitiva** com posições A1, B2, C3, etc.

## 📊 Resultados Típicos

- **Melhor Algoritmo**: MLP (Rede Neural)
- **Acurácia no Teste**: ~57% (conjunto balanceado)
- **Acurácia no Jogo**: ~75-90% (estados mais comuns)

## 🔧 Variáveis Personalizadas

Todas as variáveis seguem o padrão `nome_s` para simular `nome_$` conforme enunciado:
- `SEED_s`, `model_s`, `X_train_s`, `y_test_s`, etc.

## 👥 Autores

[Adicione seu nome aqui]

## 📝 Notas

- Implementação **100% conforme o enunciado**
- Dataset UCI original + estados gerados
- Código otimizado para reutilizar modelos treinados
- Interface clara e educativa