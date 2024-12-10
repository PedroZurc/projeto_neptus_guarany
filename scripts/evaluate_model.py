import joblib
import pandas as pd

from src.data.data_processor import DataProcessor
from src.evaluation.evaluator import Evaluator
from src.config.config import MODEL_SAVE_PATH, PROCESSED_DATA_PATH

if __name__ == "__main__":
    model = joblib.load(MODEL_SAVE_PATH)
    evaluator = Evaluator()
    
    test_df = pd.read_parquet(PROCESSED_DATA_PATH)
    X_test = test_df.drop('Venda', axis=1)
    y_test = test_df['Venda']
    
    y_pred = model.predict(X_test)
    metrics = evaluator.evaluate(y_test, y_pred)
    evaluator.save_metrics(metrics)
    
    print('Métricas salvas em:', MODEL_SAVE_PATH)
