import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config.config import METRICS_OUTPUT_PATH

class Evaluator:
    def __init__(self):
        pass
    
    def evaluate(self, y_true, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        return metrics
    
    def save_metrics(self, metrics):
        with open(METRICS_OUTPUT_PATH, 'w') as f:
            json.dump(metrics, f)