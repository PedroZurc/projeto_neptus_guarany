from src.data.data_loader import Dataloader
from src.data.data_processor import DataProcessor
from src.model.my_model import MyModel
from src.config.config import MODEL_SAVE_PATH
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model=None):
        self.model = model if model else MyModel()
        
    def plot_training_results(self, evals_result):
        iterations = range(1, len(evals_result['learn']['Logloss']) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, evals_result['learn']['Logloss'], label='Treino Logloss')
        plt.plot(iterations, evals_result['validation']['Logloss'], label='Validação Logloss')
        plt.xlabel('Iterações')
        plt.ylabel('Logloss')
        plt.title('Evolução do Logloss durante o treinamento')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def run(self):
        loader = Dataloader()
        preprocessor = DataProcessor()
        
        # Carregar e preprocessar os dados
        df = loader.load_data()
        df = preprocessor.feature_engineering(df)
        df = preprocessor.clean_data(df)
        
        # Separar variáveis independentes (X) e a variável alvo (y)
        X = df.drop('Venda', axis=1)
        y = df['Venda']
        
        # Identificar colunas categóricas pelo tipo
        cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object" or X[col].dtype.name == "category"]
        
        # Tratar valores ausentes nas colunas categóricas
        if cat_features:
            X.iloc[:, cat_features] = X.iloc[:, cat_features].fillna("missing").astype(str)
        
        # Dividir os dados em treino e validação
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Treinar o modelo com hiperparâmetros otimizados e monitoramento via conjunto de validação
        self.model.train(
            X_train, y_train, 
            cat_features=cat_features,
            eval_set=(X_valid, y_valid),      # Parâmetro adicional
            early_stopping_rounds=50,           # Parâmetro adicional
            iterations=1000,                    # Número máximo de iterações
            learning_rate=0.03,                 # Taxa de aprendizado
            depth=8                           # Profundidade da árvore
            # Outros hiperparâmetros podem ser adicionados aqui
        )
        
        # Após o treinamento, extrair os resultados para plotagem
        evals_result = self.model.get_evals_result()
        self.plot_training_results(evals_result)
        
        # Salvar o modelo treinado
        self.model.save(MODEL_SAVE_PATH)
