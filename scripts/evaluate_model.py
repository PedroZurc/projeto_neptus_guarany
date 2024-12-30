import joblib
import pandas as pd

from src.data.data_processor import DataProcessor
from src.evaluation.evaluator import Evaluator
from src.config.config import MODEL_SAVE_PATH, PROCESSED_DATA_PATH

if __name__ == "__main__":
    model = joblib.load(MODEL_SAVE_PATH)
    evaluator = Evaluator()
    
    # Carregar os dados de teste
    test_df = pd.read_excel(PROCESSED_DATA_PATH)
    X_test = test_df.drop('Venda', axis=1)
    y_test = test_df['Venda']
    
    # Tratar valores ausentes (NaN) nas colunas categóricas
    cat_features = [i for i, col in enumerate(X_test.columns) if X_test[col].dtype == "object" or X_test[col].dtype == "category"]
    
    if cat_features:  # Certifique-se de que há colunas categóricas
        X_test.iloc[:, cat_features] = X_test.iloc[:, cat_features].fillna("missing").astype(str)

    # Fazer predições
    y_pred = model.predict(X_test)
    
    # Avaliar o modelo
    metrics = evaluator.evaluate(y_test, y_pred)
    evaluator.save_metrics(metrics)
    
    print('Métricas salvas em:', MODEL_SAVE_PATH)

