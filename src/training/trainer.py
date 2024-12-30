from src.data.data_loader import Dataloader
from src.data.data_processor import DataProcessor
from src.model.my_model import MyModel
from src.config.config import MODEL_SAVE_PATH

class Trainer:
    def __init__(self, model=None):
        self.model = model if model else MyModel()
        
    def run(self):
        loader = Dataloader()
        preprocessor = DataProcessor()
        
        # Carregar e preprocessar os dados
        df = loader.load_data()
        df = preprocessor.feature_engineering(df)
        df = preprocessor.clean_data(df)
        
        # Separar variáveis independentes (X) e dependente (y)
        X = df.drop('Venda', axis=1)
        y = df['Venda']
        
        # Identificar colunas categóricas pelo tipo
        cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object" or X[col].dtype == "category"]
        
        # Tratar valores ausentes (NaN) nas colunas categóricas
        if cat_features:  # Certifique-se de que há colunas categóricas
            X.iloc[:, cat_features] = X.iloc[:, cat_features].fillna("missing").astype(str)
        
        # Treinar o modelo
        self.model.train(X, y, cat_features=cat_features)
        
        # Salvar o modelo treinado
        self.model.save(MODEL_SAVE_PATH)
