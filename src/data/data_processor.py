import pandas as pd 
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        colunas_para_remover = [
        'Emails', 'Telefones', 'CPF', 'CPF/Semponto', 'Nome', 'IdPessoa', 'CEP',
        'Cidade', 'AgenteTurismo', 'Evento', 'UH', 'Status', 
        'TarifarioHospedagem', 'NomeReservante', 'TelefoneReservante', 
        'Hotel', 'PlacaVeiculo','TiposLancamentoContaEmpresa', 'NumeroCartao', 
        'PreCheckIn', 'PossuiDescontoEspecial', 'DescontoEspecial',
        'CheckInPrevisao', 'CheckOutPrevisao'
        ]
        
        # df_clean = df.dropna()
        df_clean = df.drop(df[df['ValorTarifa'] < 150].index)
        df_clean = df_clean.drop(columns=[col for col in colunas_para_remover if col in df.columns], axis=1)
        
        return df_clean
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df['DiasReserva'] = (df['CheckOutPrevisao'] - df['CheckInPrevisao']).dt.days
        df['QuantidadeCrianca'] = df['QuantidadeCrianca1'] + df['QuantidadeCrianca2']

        df['CheckInPrevisao_month'] = df['CheckInPrevisao'].dt.month
        df['CheckInPrevisao_day'] = df['CheckInPrevisao'].dt.day
        df['CheckInPrevisao_dayofweek'] = df['CheckInPrevisao'].dt.dayofweek  # Monday=0, Sunday=6

        df['CheckOutPrevisao_month'] = df['CheckOutPrevisao'].dt.month
        df['CheckOutPrevisao_day'] = df['CheckOutPrevisao'].dt.day
        df['CheckOutPrevisao_dayofweek'] = df['CheckOutPrevisao'].dt.dayofweek  # Monday=0, Sunday=6

        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_columns) > 0:
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        
        return df
    