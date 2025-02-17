import joblib
from catboost import CatBoostClassifier
from src.model.base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, iterations=1000, learning_rate=0.01, depth=6, random_state=42):
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_state,
            verbose=False  # Para evitar muitos prints durante o treino
        )

    def train(self, X, y, cat_features=None, **kwargs):
        self.model = CatBoostClassifier(**kwargs)
        self.model.fit(X, y, cat_features=cat_features)

        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)