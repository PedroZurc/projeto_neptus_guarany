# Documentação do Modelo

## Objetivo do Modelo

O modelo busca classificar clientes potenciais para multipropriedade, retornando uma probabilidade (0% a 100%) de que aquele cliente tenha o perfil ideal.

## Modelo Utilizado

- **Algoritmo**: `CatBoostClassifier`
- **Principais Parâmetros**:
  - `iterations`: número de iterações de boosting (default: 1000)
  - `learning_rate`: taxa de aprendizado (default: 0.01)
  - `depth`: profundidade da árvore (default: 6)

Esses parâmetros podem ser ajustados no arquivo `my_model.py`.

## Dados de Entrada

- Colunas categóricas e numéricas.
- As colunas categóricas são mantidas como tipo `object` ou `category`.
- As colunas numéricas são escaladas com `StandardScaler`.

## Saída do Modelo

- Probabilidade de pertencer à classe "perfil ideal" (posteriormente convertida em porcentagem).
- Predição final da classe (0 ou 1).

## Futuras Extensões

- Incluir mais features após engenharia de características.
- Testar outros modelos (XGBoost, LightGBM) criando novas classes que implementem `BaseModel`.
- Ajustar hiperparâmetros via `GridSearchCV` ou `Optuna` para melhorar a performance.

