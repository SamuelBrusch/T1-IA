# Jogo da Velha com IA Classificadora

Trabalho de Inteligência Artificial.

Projeto em Python que treina 4 algoritmos (k-NN, MLP, Árvore de Decisão, Random Forest) para classificar o estado do Jogo da Velha em tempo real durante partidas Humano vs IA. O jogador escolhe qual algoritmo enfrentar; a IA faz jogadas aleatórias.

## Requisitos

- Python 3.10+
- Windows PowerShell (os comandos abaixo usam PowerShell)

## Como rodar (Windows)

1) Ative o ambiente virtual e instale as dependências:

```
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) (Opcional) Treine/re-treine os modelos:

```
python main.py --train
```

3) Rode o jogo (treina automaticamente se não houver modelo):

```
python main.py
```

Na abertura, selecione o algoritmo (1-4). No tabuleiro, use posições como A1, B2, C3. Digite "sair" para encerrar.

## Estrutura do Projeto

```
T1 IA/
├── src/
│   ├── config.py            # Caminhos e pastas
│   ├── data_and_train.py    # Geração do dataset e treinamento
│   └── game.py              # Interface console Humano vs IA
├── models/
│   ├── best_model.pkl       # Melhor modelo global
│   ├── scaler.pkl           # Scaler do melhor modelo (se houver)
│   ├── model_metadata.pkl   # Metadados
│   ├── knn_model.pkl        # Modelos por algoritmo
│   ├── mlp_model.pkl
│   ├── tree_model.pkl
│   └── forest_model.pkl
├── data/
│   └── algorithm_comparison.csv  # Resultados de validação
├── main.py
├── requirements.txt
└── README.md
```

## Observações

- O jogo é Humano (X) vs IA (O). A IA joga aleatoriamente; a classificação do estado ("X vence", "O vence", "Empate", "Tem jogo", "Possibilidade de Fim de Jogo") é feita pelo modelo escolhido e exibida a cada turno com estatísticas de acerto.
- Se um modelo específico não existir, o programa pode treinar automaticamente.
- Para recomeçar, basta rodar novamente com `python main.py`.