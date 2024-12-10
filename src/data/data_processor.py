import pandas as pd 
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        colunas_para_remover = [
        'Emails', 'Telefones', 'CPF', 'Nome', 'IdPessoa', 'CEP',
        'Cidade', 'AgenteTurismo', 'Evento', 'UH', 'Status', 
        'TarifarioHospedagem', 'NomeReservante', 'TelefoneReservante', 
        'Hotel', 'PlacaVeiculo','TiposLancamentoContaEmpresa', 'NumeroCartao', 
        'PreCheckIn', 'PossuiDescontoEspecial', 'DescontoEspecial',
        'CheckInPrevisao', 'CheckOutPrevisao'
        ]
        
        df_clean = df.dropna()
        df_clean = df_clean.drop(df_clean[df_clean['ValorTarifa'] < 150].index)

        return df_clean
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df['DiasReserva'] = (df['CheckOutPrevisao'] - df['CheckInPrevisao']).dt.days
        df['QuantidadeCrianca'] = (df['QuantidadeCrianca1'] + df['QuantidadeCrianca2'])
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_columns) > 0:
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        
        return df