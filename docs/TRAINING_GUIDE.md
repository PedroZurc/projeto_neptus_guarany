# Guia de Treinamento

## Pré-requisitos

- Python 3.8+
- Dependências instaladas via `requirements.txt`.

## Passos para Treinamento

1. **Preparar dados**:  
   - Coloque o arquivo `dados_treinamento.xlsx` em `data/raw/`.
   
2. **Executar Treinamento**:  
   - Rode o script de treinamento:
     ```bash
     python scripts/train_model.py
     ```
     
   Isso irá:
   - Carregar os dados.
   - Pré-processar (limpeza, feature engineering, escala numéricos, manter categóricos).
   - Treinar o modelo `CatBoostClassifier`.
   - Salvar o modelo em `project/models/model.pkl` ou no formato nativo do CatBoost.

## Ajuste de Parâmetros

- Ajuste parâmetros em `my_model.py` (por exemplo, `iterations`, `learning_rate`, `depth`).
- Refaça o treinamento após ajustes para avaliar melhorias.

## Utilizando Outras Fontes de Dados

- Caso queira usar outro arquivo de dados, altere `DATA_PATH` em `config.py`.
- Atualize o fluxo no `Trainer` se necessário.

## Troubleshooting

- Erros de path: verifique se o `DATA_PATH` está correto em `config.py`.
- Erros de falta de pacotes: rode novamente `pip install -r requirements.txt`.
- Erros relacionados a tipos de dados: verifique se as colunas categóricas e numéricas estão corretamente definidas no pré-processamento.

