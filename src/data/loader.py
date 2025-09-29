import pandas as pd
from zipfile import ZipFile
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
    
    def create_basic_education_table(self):
        descricao_campos_evasao = [
            'Nome da Unidade da Federação',
            'Nome do Município',

            # Perfil da Escola
            'Dependência Administrativa', # Porcentagem
            #'Localização',
            #'Situação de funcionamento',
            'Prédio compartilhado com outra escola', # Porcentagem
            'Escola abre aos finais de semana para a comunidade', # Porcentagem
            'A escola faz exame de seleção para ingresso de seus alunos (avaliação por prova e/ou análise curricular)', # Porcentagem NAO PEGOU

            # Infraestrutura
            'Abastecimento de água - Não há abastecimento de água', # Porcentagem
            'Abastecimento de energia elétrica - Não há energia elétrica', # Porcentagem
            'Esgoto sanitário - Não há esgotamento sanitário', # Porcentagem
            'Dependências físicas existentes e utilizadas na escola - Biblioteca', # Porcentagem
            'Dependências físicas existentes e utilizadas na escola - Laboratório de informática', # Porcentagem
            'Dependências físicas existentes e utilizadas na escola - Pátio Coberto', # Porcentagem
            'Dependências físicas existentes e utilizadas na escola - Quadra de esportes coberta ou descoberta', # Porcentagem
            'Dependências físicas existentes e utilizadas na escola - Refeitório', # Porcentagem
            'Número de salas de aula existentes na escola', # Média, desvio padrão, min e max
            'Acesso à Internet', # Porcentagem
            'Acesso à Internet - Para uso dos alunos', # Porcentagem
            'Internet Banda Larga', # Porcentagem

            # Recursos Humanos e Corpo Docente
            'Total de funcionários da escola (inclusive profissionais escolares em sala de aula)',# Média, desvio padrão, min e max
            'Número de Docentes do Ensino Médio', # Média, desvio padrão, min e max
            'Profissionais que atuam na escola - Psicólogo(a) Escolar', # Porcentagem
            'Profissionais que atuam na escola - Orientador(a) comunitário(a) ou assistente social', # Porcentagem

            # Perfil dos Alunos (Dados Agregados)
            'Número de Matrículas no Ensino Médio',
            'Número de Matrículas no Ensino Médio - Tempo Integral',
            'Órgãos colegiados em funcionamento na escola - Grêmio Estudantil',
        ]
        
        # Read the dictionary file
        dicionario_microdados = pd.read_excel('data/dicionario_dados_educacao_basica.xlsx', skiprows=6)
        
        # First, let's read a sample of the CSV to get actual column names
        with ZipFile('data/microdados_ed_basica_2024.zip') as z:
            with z.open('microdados_ed_basica_2024.csv') as f:
                # Read just the header to get column names
                sample = pd.read_csv(f, sep=';', encoding='latin1', nrows=0)
                available_columns = sample.columns.tolist()
    
        # Filter campos_relevantes_evasao to only include columns that exist in the CSV
        campos_relevantes_evasao = (dicionario_microdados[
            dicionario_microdados['Descrição da Variável'].isin(descricao_campos_evasao)]
            .iloc[:, 1].tolist())
    
        campos_existentes = [col for col in campos_relevantes_evasao if col in available_columns]
    
        # 'Escola abre aos finais de semana para a comunidade', # Porcentagem NAO PEGOU
        # 'A escola faz exame de seleção para ingresso de seus alunos (avaliação por prova e/ou análise curricular)', # Porcentagem NAO PEGOU
        # 'Número de salas de aula existentes na escola', # Média, desvio padrão, min e max  NAO PEGOU
        # 'Total de funcionários da escola (inclusive profissionais escolares em sala de aula)',# Média, desvio padrão, min e max NAO PEGOU
        # 'Número de Matrículas no Ensino Médio', NAO PEGOU
        # 'Número de Matrículas no Ensino Médio - Tempo Integral', NAO PEGOU
        
        # 'NO_UF'
        # 'NO_MUNICIPIO'
        # 'TP_DEPENDENCIA'
        # 'IN_PREDIO_COMPARTILHADO'
        # 'IN_AGUA_INEXISTENTE'
        # 'IN_ENERGIA_INEXISTENTE'
        # 'IN_ESGOTO_INEXISTENTE'
        # 'IN_BIBLIOTECA'
        # 'IN_LABORATORIO_INFORMATICA'
        # 'IN_QUADRA_ESPORTES'
        # 'IN_REFEITORIO'
        # 'IN_INTERNET'
        # 'IN_INTERNET_ALUNOS'
        # 'IN_BANDA_LARGA'
        # 'IN_PROF_PSICOLOGO'
        # 'IN_PROF_ASSIST_SOCIAL'
        # 'IN_EXAME_SELECAO'
        # 'IN_ORGAO_GREMIO_ESTUDANTIL'
        # 'QT_DOC_MED'
        
        
        print(dicionario_microdados['Descrição da Variável'].to_list())
        print('---'*40)
        print("Colunas disponíveis:", available_columns)
        print('---'*40)
        print("\nColunas selecionadas:", campos_existentes)
    
        # Now read the CSV with only the existing columns
        with ZipFile('data/microdados_ed_basica_2024.zip') as z:
            with z.open('microdados_ed_basica_2024.csv') as f:
                microdados = pd.read_csv(f, sep=';', encoding='latin1', 
                                       usecols=campos_existentes)
                microdados = microdados.fillna(microdados.mean(numeric_only=True))
    
        # Continue with the rest of the processing
        microdados['NO_UF'] = microdados['NO_UF'].map(estados_br)
    
        df_municipio = (microdados[microdados['TP_DEPENDENCIA'] != 4]
                       .drop(columns=['TP_DEPENDENCIA'])
                       .groupby(['NO_UF', 'NO_MUNICIPIO'])
                       .agg("mean")
                       .reset_index())
    
        return df_municipio
    
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