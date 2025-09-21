import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self, filepath, skiprows=8):
        self.filepath = filepath
        self.skiprows = skiprows
        self.scaler = MinMaxScaler()

    def load_data(self):
        df = pd.read_excel(
            self.filepath,
            skiprows=self.skiprows,
            usecols=[1,2,4,5,6,51,52,53,54]
        )
        
        df = self._rename_columns(df)
        df = self._clean_data(df)
        return df

    def _rename_columns(self, df):
        return df.rename(columns={
            '1_CAT3_CATMED': 'tx_evasao_total_EM',
            '1_CAT3_CATMED_01': 'tx_evasao_1_ano_EM',
            '1_CAT3_CATMED_02': 'tx_evasao_2_ano_EM',
            '1_CAT3_CATMED_03': 'tx_evasao_3_ano_EM'
        })

    def _clean_data(self, df):
        df = df.query("NO_LOCALIZACAO == 'Total' and NO_DEPENDENCIA == 'Pública'")
        df = df.drop(columns=['NO_LOCALIZACAO', 'NO_DEPENDENCIA'])
        
        colunas_interesse = ['tx_evasao_total_EM', 'tx_evasao_1_ano_EM', 
                           'tx_evasao_2_ano_EM', 'tx_evasao_3_ano_EM']
        
        for col in colunas_interesse:
            df = df[(df[col] != '--') & (df[col] != '***')]
        
        df[colunas_interesse] = df[colunas_interesse].astype(float)
        df = df.dropna(subset=['tx_evasao_total_EM'])
        df = df.fillna(df.mean(numeric_only=True))
        
        return df

    def prepare_data(self, df, test_size=0.2, random_state=42):
        X = df[['tx_evasao_1_ano_EM', 'tx_evasao_2_ano_EM', 'tx_evasao_3_ano_EM']]
        y = df['tx_evasao_total_EM']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test