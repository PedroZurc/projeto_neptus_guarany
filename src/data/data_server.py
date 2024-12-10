# src/data/data_saver.py
import pandas as pd
import os

class DataSaver:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_parquet(self, df: pd.DataFrame, file_name: str):
        output_path = os.path.join(self.output_dir, file_name)
        df.to_parquet(output_path, index=False)
        print(f"DataFrame salvo em {output_path}")
