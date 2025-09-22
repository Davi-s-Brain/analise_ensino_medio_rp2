import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

estados_br = {
    "Acre": "AC",
    "Alagoas": "AL",
    "Amapá": "AP",
    "Amazonas": "AM",
    "Bahia": "BA",
    "Ceará": "CE",
    "Distrito Federal": "DF",
    "Espírito Santo": "ES",
    "Goiás": "GO",
    "Maranhão": "MA",
    "Mato Grosso": "MT",
    "Mato Grosso do Sul": "MS",
    "Minas Gerais": "MG",
    "Pará": "PA",
    "Paraíba": "PB",
    "Paraná": "PR",
    "Pernambuco": "PE",
    "Piauí": "PI",
    "Rio de Janeiro": "RJ",
    "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS",
    "Rondônia": "RO",
    "Roraima": "RR",
    "Santa Catarina": "SC",
    "São Paulo": "SP",
    "Sergipe": "SE",
    "Tocantins": "TO"
}

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
        # # Remover outliers
        df = self._remove_outliers(df)

        # # Normalizar dados
        # df = self._normalize_features(df)

        X = df[['NO_REGIAO', 'NO_UF', 'NO_MUNICIPIO', 'tx_evasao_1_ano_EM', 'tx_evasao_2_ano_EM', 'tx_evasao_3_ano_EM', 'MEDIA_INSE', 'PC_NIVEL_1', 'PC_NIVEL_2', 'PC_NIVEL_3', 'PC_NIVEL_4', 'PC_NIVEL_5', 'PC_NIVEL_6', 'PC_NIVEL_7']]
        y = df['tx_evasao_total_EM']
        
        X = pd.get_dummies(X, columns=['NO_REGIAO', 'NO_UF', 'NO_MUNICIPIO'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_inse_table(self):
        inse = (pd.read_excel('data/INSE_2019_MUNICIPIOS.xlsx', skiprows=2, usecols=[1,3,4,5,7,8,9,10,11,12,13,14]).rename(columns={'NOME_UF': 'NO_UF', 'NOME_MUNICIPIO': 'NO_MUNICIPIO', 'MEDIA_INSE**': 'MEDIA_INSE'}))
        
        inse = (inse[(inse['TP_TIPO_REDE'] == 6)
             & (inse['TP_LOCALIZACAO'] == 0)]
             .drop(columns=['TP_TIPO_REDE', 'TP_LOCALIZACAO'])
             )
        
        inse['NO_UF'] = inse['NO_UF'].map(estados_br)
        
        return inse
    
    def combine_data(self, df_inep, df_inse):
        df_combined = pd.merge(df_inep, df_inse, on=['NO_UF', 'NO_MUNICIPIO'], how='left').dropna()
        return df_combined
    
    def _remove_outliers(self, df, n_std=3):
        columns = ['tx_evasao_total_EM', 'tx_evasao_1_ano_EM', 
                   'tx_evasao_2_ano_EM', 'tx_evasao_3_ano_EM', 'MEDIA_INSE']
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] <= mean + (n_std * std)) & 
                    (df[col] >= mean - (n_std * std))]
        return df