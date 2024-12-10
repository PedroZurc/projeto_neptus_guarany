# Métricas do Modelo

## Métricas Calculadas

- **Accuracy**: Percentual de acertos do modelo sobre o total de previsões.
- **Precision**: Entre as previsões positivas, quantas estão corretas.
- **Recall**: Entre os casos positivos reais, quantos foram identificados corretamente.
- **F1 Score**: Métrica harmônica entre Precision e Recall, equilibrando ambas.

## Interpretação das Métricas

- **Accuracy**: Boa para ter uma visão geral, mas pode ser enganosa se há desbalanceamento entre classes.
- **Precision**: Se alta, significa poucos falsos positivos (bom para evitar ofertar a clientes com pouco potencial).
- **Recall**: Se alta, significa poucos falsos negativos (bom para não perder clientes potenciais).
- **F1**: Indicado quando se quer um equilíbrio entre precisão e recall.

## Salvando Métricas

As métricas após avaliação são salvas em `project/results/metrics.json`.

Exemplo de formato:

```json
{
    "accuracy": 0.85,
    "precision": 0.80,
    "recall": 0.78,
    "f1": 0.79
}
