from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.model.my_model import MyModel
from src.config.config import MODEL_SAVE_PATH

class Trainer:
    def __init__(self, model=None):
        self.model = model if model else MyModel()
        
    def run(self):
        loader = DataLoader()
        preprocessor = DataProcessor()
        
        df = loader.load_data()
        df = preprocessor.clean_data(df)
        df = preprocessor.feature_engineering(df)
        
        X = df.drop('Venda', axis=1)
        y = df['Venda']
        
        cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object" or X[col].dtype == "category"]
        
        self.model.train(X, y, cat_features=cat_features)
        self.model.save(MODEL_SAVE_PATH)