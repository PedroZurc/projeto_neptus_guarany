from src.data.data_loader import Dataloader
from src.data.data_processor import DataProcessor
from src.data.data_server import DataSaver
from src.config.config import PROCESSED_DATA_PATH

def main():
    loader = Dataloader()
    df = loader.load_data()

    preprocessor = DataProcessor()
    df = preprocessor.feature_engineering(df)
    df = preprocessor.clean_data(df)
    # df = preprocessor.scale_features(df)

    # 3. Salva o dado preprocessado em parquet
    # Extrai o diretório do PROCESSED_DATA_PATH para inicializar o DataSaver
    # import os
    # output_dir = os.path.dirname(PROCESSED_DATA_PATH)
    # file_name = os.path.basename(PROCESSED_DATA_PATH)

    # saver = DataSaver(output_dir=output_dir)
    # saver.save_parquet(df, file_name)
    
    df.to_excel('data/processed/dados_tratados.xlsx', index=False)

    print("Dados pré-processados salvos com sucesso!")

if __name__ == "__main__":
    main()
