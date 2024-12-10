import pandas as pd
from src.config.config import DATA_PATH

class Dataloader():
    def __init__(self):
        self.data_path = DATA_PATH
        
    def load_data(self) -> pd.DataFrame:
        return pd.read_excel(self.data_path)