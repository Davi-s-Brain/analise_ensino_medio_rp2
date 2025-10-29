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

    def prepare_data(self, df, test_size=0.2, random_state=42):
        # # Remover outliers
        df = self._remove_outliers(df)

        # # Normalizar dados
        # df = self._normalize_features(df)
        
        print(df.columns.to_list())

        X = df[['NO_REGIAO', 'NO_UF',
                
                # INSE
                'MEDIA_INSE', 'PC_NIVEL_1', 'PC_NIVEL_2', 'PC_NIVEL_3', 'PC_NIVEL_4', 'PC_NIVEL_5', 'PC_NIVEL_6', 'PC_NIVEL_7',

                # Microdados
                'QT_DOC_MED_mean', 'QT_DOC_MED_std', 'QT_DOC_MED_min', 'QT_DOC_MED_max', 'QT_SALAS_EXISTENTES_mean', 'QT_SALAS_EXISTENTES_std', 'QT_SALAS_EXISTENTES_min', 'QT_SALAS_EXISTENTES_max', 'QT_FUNCIONARIOS_mean', 'QT_FUNCIONARIOS_std', 'QT_FUNCIONARIOS_min', 'QT_FUNCIONARIOS_max', 'IN_PREDIO_COMPARTILHADO_mean', 'IN_AGUA_INEXISTENTE_mean', 'IN_ENERGIA_INEXISTENTE_mean', 'IN_ESGOTO_INEXISTENTE_mean', 'IN_BIBLIOTECA_mean', 'IN_LABORATORIO_INFORMATICA_mean', 'IN_QUADRA_ESPORTES_mean', 'IN_REFEITORIO_mean', 'IN_INTERNET_mean', 'IN_INTERNET_ALUNOS_mean', 'IN_BANDA_LARGA_mean', 'IN_PROF_PSICOLOGO_mean', 'IN_PROF_ASSIST_SOCIAL_mean', 'IN_EXAME_SELECAO_mean', 'IN_ORGAO_GREMIO_ESTUDANTIL_mean', 'IN_FINAL_SEMANA_mean', 'QT_MAT_MED_sum', 'QT_MAT_MED_INT_sum',
                
                # Adequação da formação docente (AFD)
                'MED_CAT_1_afd', 'MED_CAT_2_afd', 'MED_CAT_3_afd', 'MED_CAT_4_afd', 'MED_CAT_5_afd',

                # Indicador de esforço docente (IED)
                'MED_CAT_1_ied', 'MED_CAT_2_ied', 'MED_CAT_3_ied', 'MED_CAT_4_ied', 'MED_CAT_5_ied', 'MED_CAT_6_ied',

                # Índice de Desenvolvimento da Educação Básica (Ideb)
                'VL_APROVACAO_2021_SI_4', 'VL_NOTA_MATEMATICA_2021', 'VL_NOTA_PORTUGUES_2021', 'VL_NOTA_MEDIA_2021', 'VL_OBSERVADO_2021', 'VL_PROJECAO_2021',

                # Média de alunos por turma (ATU)
                'MED_CAT_0_atu',

                # Média Horas-aula diária (HAD)
                'MED_CAT_0_had',

                # Percentual de docentes com curso superior (DSU)
                'MED_CAT_0_dsu',

                # Regularidade do corpo docente (IRD)
                'EDU_BAS_CAT_1', 'EDU_BAS_CAT_2', 'EDU_BAS_CAT_3', 'EDU_BAS_CAT_4',

                # Taxa de distorção idade série (TDI)
                'MED_CAT_0_tdi', #, 'MED_01_CAT_0_tdi', 'MED_02_CAT_0_tdi', 'MED_03_CAT_0_tdi'

                # Remuneração média dos docentes (RMD)
                'ED_BAS_CAT1', 'ED_BAS_CAT2', 'ED_BAS_CAT3', 'ED_BAS_CAT4', 'ED_BAS_CAT5', 'ED_BAS_CAT6', 'ED_BAS_CAT7', 'ED_BAS_CAT8',
                
                # Taxa de Não Resposta (TNR) - parece não fazer diferença
                #"4_CAT_MED_tnr"
                
                # Taxa de rendimento
                'tx_aprovacao_EM', 'tx_reprovacao_EM', 'tx_abandono_EM'

                ]]
        
        y = df['tx_evasao_total_EM']
        
        X = pd.get_dummies(X, columns=['NO_REGIAO', 'NO_UF'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    

    def create_transicao_table(self):
        df_transicao = (
        pd.read_excel('data/TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx', skiprows=8, usecols=[1,2,4,5,6,19,35,51,52,53,54,67])

        .rename(columns={
                '1_CAT3_CATMED': 'tx_evasao_total_EM',
                '1_CAT3_CATMED_01': 'tx_evasao_1_ano_EM',
                '1_CAT3_CATMED_02': 'tx_evasao_2_ano_EM',
                '1_CAT3_CATMED_03': 'tx_evasao_3_ano_EM',
                '1_CAT1_CATMED': 'tx_promocao_EM',
                '1_CAT2_CATMED': 'tx_repetencia_EM',
                '1_CAT4_CATMED': 'tx_migracao_eja_EM'

            })

        .query("NO_LOCALIZACAO == 'Total' and NO_DEPENDENCIA == 'Pública'")
        .drop(columns=['NO_LOCALIZACAO', 'NO_DEPENDENCIA'])
        )

        colunas_interesse = ['tx_promocao_EM', 'tx_repetencia_EM', 'tx_evasao_total_EM', 'tx_evasao_1_ano_EM', 'tx_evasao_2_ano_EM', 'tx_evasao_3_ano_EM', 'tx_migracao_eja_EM']

        for col in colunas_interesse:
            df_transicao = df_transicao[(df_transicao[col] != '--') & (df_transicao[col] != '***')]

        df_transicao[colunas_interesse] = df_transicao[colunas_interesse].astype(float)
        df_transicao = df_transicao.dropna(subset=['tx_evasao_total_EM'])
        df_transicao = df_transicao.fillna(df_transicao.mean(numeric_only=True))
        return df_transicao



    def create_inse_table(self):
        inse = (pd.read_excel('data/INSE_2019_MUNICIPIOS.xlsx', skiprows=2, usecols=[1,3,4,5,7,8,9,10,11,12,13,14]).rename(columns={'NOME_UF': 'NO_UF', 'NOME_MUNICIPIO': 'NO_MUNICIPIO', 'MEDIA_INSE**': 'MEDIA_INSE'}))
        
        inse = (inse[(inse['TP_TIPO_REDE'] == 6) & (inse['TP_LOCALIZACAO'] == 0)]
             .drop(columns=['TP_TIPO_REDE', 'TP_LOCALIZACAO'])
             )
        
        inse['NO_UF'] = inse['NO_UF'].map(estados_br)
        
        return inse
    
    def create_basic_education_table(self):
        campos_interesse = [     
            'NO_UF',
            'NO_MUNICIPIO',
            'TP_DEPENDENCIA',
            'IN_PREDIO_COMPARTILHADO',
            'IN_AGUA_INEXISTENTE',
            'IN_ENERGIA_INEXISTENTE',
            'IN_ESGOTO_INEXISTENTE',
            'IN_BIBLIOTECA',
            'IN_LABORATORIO_INFORMATICA',
            'IN_QUADRA_ESPORTES',
            'IN_REFEITORIO',
            'IN_INTERNET',
            'IN_INTERNET_ALUNOS',
            'IN_BANDA_LARGA',
            'IN_PROF_PSICOLOGO',
            'IN_PROF_ASSIST_SOCIAL',
            'IN_EXAME_SELECAO',
            'IN_ORGAO_GREMIO_ESTUDANTIL',
            'QT_DOC_MED', # Média, desvio padrão, min e max
            'IN_FINAL_SEMANA',
            'QT_SALAS_EXISTENTES', # Média, desvio padrão, min e max
            'QT_FUNCIONARIOS', # Média, desvio padrão, min e max
            'QT_MAT_MED',
            'QT_MAT_MED_INT'
        ]
        
        dicionario_microdados = pd.read_excel('data/dicionario_dados_educacao_basica.xlsx', skiprows=6)
        
        with ZipFile('data/microdados_censo_escolar_2020.zip') as z:
            with z.open('microdados_ed_basica_2020/dados/microdados_ed_basica_2020.CSV') as f:
                sample = pd.read_csv(f, sep=';', encoding='latin1', nrows=0)
                available_columns = sample.columns.tolist()

        campos_relevantes_evasao = (dicionario_microdados[
            dicionario_microdados['Nome da Variável'].isin(campos_interesse)]
            .iloc[:, 1].tolist())

        campos_existentes = [col for col in campos_relevantes_evasao if col in available_columns]

        with ZipFile('data/microdados_censo_escolar_2020.zip') as z:
            with z.open('microdados_ed_basica_2020/dados/microdados_ed_basica_2020.CSV') as f:
                microdados = pd.read_csv(f, sep=';', encoding='latin1', usecols=campos_existentes)
                microdados = microdados.fillna(microdados.mean(numeric_only=True))
    
        microdados['NO_UF'] = microdados['NO_UF'].map(estados_br)
    
        df_municipio = (microdados[microdados['TP_DEPENDENCIA'] != 4]
                       .drop(columns=['TP_DEPENDENCIA'])
                       .groupby(['NO_UF', 'NO_MUNICIPIO'])
                       .agg({
                            'QT_DOC_MED': ['mean', 'std', 'min', 'max'],
                            'QT_SALAS_EXISTENTES': ['mean', 'std', 'min', 'max'],
                            'QT_FUNCIONARIOS': ['mean', 'std', 'min', 'max'],
                            'IN_PREDIO_COMPARTILHADO': 'mean',
                            'IN_AGUA_INEXISTENTE': 'mean',
                            'IN_ENERGIA_INEXISTENTE': 'mean',
                            'IN_ESGOTO_INEXISTENTE': 'mean',
                            'IN_BIBLIOTECA': 'mean',
                            'IN_LABORATORIO_INFORMATICA': 'mean',
                            'IN_QUADRA_ESPORTES': 'mean',
                            'IN_REFEITORIO': 'mean',
                            'IN_INTERNET': 'mean',
                            'IN_INTERNET_ALUNOS': 'mean',
                            'IN_BANDA_LARGA': 'mean',
                            'IN_PROF_PSICOLOGO': 'mean',
                            'IN_PROF_ASSIST_SOCIAL': 'mean',
                            'IN_EXAME_SELECAO': 'mean',
                            'IN_ORGAO_GREMIO_ESTUDANTIL': 'mean',
                            'IN_FINAL_SEMANA': 'mean',
                            'QT_MAT_MED': 'sum',
                            'QT_MAT_MED_INT': 'sum'
                        })
                       .reset_index())

        df_municipio.columns = [
            '_'.join(col).strip() if isinstance(col, tuple) else col 
            for col in df_municipio.columns.values
        ]
        
        df_municipio = df_municipio.rename(columns=lambda x: x.rstrip('_') if x.endswith('_') else x)
        
        return df_municipio
    




    def create_afd_table(self):
        df_afd = pd.read_excel('data/AFD_MUNICIPIOS_2021.xlsx', skiprows=10, usecols=[2,4,5,6,27,28,29,30,31])
        df_afd = (
            df_afd.query("NO_DEPENDENCIA == 'Pública' and NO_CATEGORIA == 'Total'")
            .drop(columns=['NO_DEPENDENCIA', 'NO_CATEGORIA'])
            .rename(columns={'SG_UF': 'NO_UF',
                             'MED_CAT_0': 'MED_CAT_0_afd',
                             'MED_CAT_1': 'MED_CAT_1_afd',
                             'MED_CAT_2': 'MED_CAT_2_afd',
                             'MED_CAT_3': 'MED_CAT_3_afd',
                             'MED_CAT_4': 'MED_CAT_4_afd',
                             'MED_CAT_5': 'MED_CAT_5_afd'
                             })
        )
        colunas_interesse = ['MED_CAT_1_afd', 'MED_CAT_2_afd', 'MED_CAT_3_afd', 'MED_CAT_4_afd', 'MED_CAT_5_afd']
        for col in colunas_interesse:
            df_afd = df_afd[(df_afd[col] != '--') & (df_afd[col] != '***')]
        df_afd[colunas_interesse] = df_afd[colunas_interesse].astype(float)
        df_afd = df_afd.fillna(df_afd.mean(numeric_only=True))

        return df_afd


    def create_ied_table(self):
        df_ied = pd.read_excel('data/IED_MUNICIPIOS_2021.xlsx', skiprows=10, usecols=[2,4,5,6,25,26,27,28,29,30])
        df_ied = (
            df_ied.query("NO_DEPENDENCIA == 'Pública' and NO_CATEGORIA == 'Total'")
            .drop(columns=['NO_DEPENDENCIA', 'NO_CATEGORIA'])
            .rename(columns={'SG_UF': 'NO_UF',
                             'MED_CAT_0': 'MED_CAT_0_ied',
                             'MED_CAT_1': 'MED_CAT_1_ied',
                             'MED_CAT_2': 'MED_CAT_2_ied',
                             'MED_CAT_3': 'MED_CAT_3_ied',
                             'MED_CAT_4': 'MED_CAT_4_ied',
                             'MED_CAT_5': 'MED_CAT_5_ied',
                             'MED_CAT_6': 'MED_CAT_6_ied'
                             })
        )
        colunas_interesse = ['MED_CAT_1_ied', 'MED_CAT_2_ied', 'MED_CAT_3_ied', 'MED_CAT_4_ied', 'MED_CAT_5_ied', 'MED_CAT_6_ied']
        for col in colunas_interesse:
            df_ied = df_ied[(df_ied[col] != '--') & (df_ied[col] != '***')]
        df_ied[colunas_interesse] = df_ied[colunas_interesse].astype(float)
        df_ied = df_ied.fillna(df_ied.mean(numeric_only=True))
        return df_ied

    def create_ideb_table(self):
        df_ideb = pd.read_excel('data/IDEB_MUNICIPIOS_2023.xlsx', skiprows=9, usecols=[0,2,3,16,34,35,36,42,45])
        df_ideb = (
            df_ideb.query("REDE == 'Pública'")
            .drop(columns=['REDE'])
            .rename(columns={'SG_UF': 'NO_UF'})
        )
        colunas_interesse = [
            'VL_APROVACAO_2021_SI_4', 'VL_NOTA_MATEMATICA_2021',
            'VL_NOTA_PORTUGUES_2021', 'VL_NOTA_MEDIA_2021',
            'VL_OBSERVADO_2021', 'VL_PROJECAO_2021'
        ]
        for col in colunas_interesse:
            df_ideb = df_ideb[(df_ideb[col] != '-') & (df_ideb[col] != '***')]
        df_ideb[colunas_interesse] = df_ideb[colunas_interesse].astype(float)
        df_ideb = df_ideb.fillna(df_ideb.mean(numeric_only=True))

        return df_ideb


    def create_atu_table(self):
        df_atu = pd.read_excel('data/ATU_MUNICIPIOS_2021.xlsx', skiprows=8, usecols=[2,4,5,6,23])
        df_atu = (
            df_atu.query("NO_DEPENDENCIA == 'Pública' and NO_CATEGORIA == 'Total'")
            .drop(columns=['NO_DEPENDENCIA', 'NO_CATEGORIA'])
            .rename(columns={'SG_UF': 'NO_UF', 'MED_CAT_0': 'MED_CAT_0_atu'})
        )
        colunas_interesse = ['MED_CAT_0_atu']
        for col in colunas_interesse:
            df_atu = df_atu[(df_atu[col] != '--') & (df_atu[col] != '***')]
        df_atu[colunas_interesse] = df_atu[colunas_interesse].astype(float)
        df_atu = df_atu.fillna(df_atu.mean(numeric_only=True))

        return df_atu

    def create_had_table(self):
        df_had = pd.read_excel('data/HAD_MUNICIPIOS_2021.xlsx', skiprows=8, usecols=[2,4,5,6,22])
        df_had = (
            df_had.query("NO_DEPENDENCIA == 'Pública' and NO_CATEGORIA == 'Total'")
            .drop(columns=['NO_DEPENDENCIA', 'NO_CATEGORIA'])
            .rename(columns={'SG_UF': 'NO_UF', 'MED_CAT_0': 'MED_CAT_0_had'})
        )
        colunas_interesse = ['MED_CAT_0_had']
        for col in colunas_interesse:
            df_had = df_had[(df_had[col] != '--') & (df_had[col] != '***')]
        df_had[colunas_interesse] = df_had[colunas_interesse].astype(float)
        df_had = df_had.fillna(df_had.mean(numeric_only=True))

        return df_had


    def create_dsu_table(self):
        df_dsu = pd.read_excel('data/DSU_MUNICIPIOS_2021.xlsx', skiprows=9, usecols=[2,4,5,6,7,13])
        df_dsu = (
            df_dsu.query("NO_CATEGORIA == 'Total' and NO_DEPENDENCIA == 'Pública'")
            .drop(columns=['NO_DEPENDENCIA', 'NO_CATEGORIA'])
            .rename(columns={'SG_UF': 'NO_UF', 'MED_CAT_0': 'MED_CAT_0_dsu'})
        )
        colunas_interesse = ['MED_CAT_0_dsu']
        for col in colunas_interesse:
            df_dsu = df_dsu[(df_dsu[col] != '--') & (df_dsu[col] != '***')]
        df_dsu[colunas_interesse] = df_dsu[colunas_interesse].astype(float)
        df_dsu = df_dsu.fillna(df_dsu.mean(numeric_only=True))
        
        return df_dsu


    def create_ird_table(self):
        df_ird = pd.read_excel('data/IRD_MUNICIPIOS_2021.xlsx', skiprows=9, usecols=[2,4,5,6,7,8,9,10])
        df_ird = (
            df_ird.query("NO_DEPENDENCIA == 'Pública' and NO_CATEGORIA == 'Total'")
            .drop(columns=['NO_DEPENDENCIA', 'NO_CATEGORIA'])
            .rename(columns={'SG_UF': 'NO_UF'})
        )
        colunas_interesse = ['EDU_BAS_CAT_1', 'EDU_BAS_CAT_2', 'EDU_BAS_CAT_3', 'EDU_BAS_CAT_4']
        for col in colunas_interesse:
            df_ird = df_ird[(df_ird[col] != '--') & (df_ird[col] != '***')]
        df_ird[colunas_interesse] = df_ird[colunas_interesse].astype(float)
        df_ird = df_ird.fillna(df_ird.mean(numeric_only=True))

        return df_ird


    def create_tdi_table(self):
        df_tdi = pd.read_excel('data/TDI_MUNICIPIOS_2021.xlsx', skiprows=8, usecols=[2,4,5,6,19,20,21,22])
        df_tdi = (
            df_tdi.query("NO_DEPENDENCIA == 'Pública' and NO_CATEGORIA == 'Total'")
            .drop(columns=['NO_DEPENDENCIA', 'NO_CATEGORIA'])
            .rename(columns={'SG_UF': 'NO_UF',
                            'MED_CAT_0': 'MED_CAT_0_tdi',
                            'MED_01_CAT_0': 'MED_01_CAT_0_tdi',
                            'MED_02_CAT_0': 'MED_02_CAT_0_tdi',
                            'MED_03_CAT_0': 'MED_03_CAT_0_tdi'})
        )
        colunas_interesse = ['MED_CAT_0_tdi', 'MED_01_CAT_0_tdi', 'MED_02_CAT_0_tdi', 'MED_03_CAT_0_tdi']
        for col in colunas_interesse:
            df_tdi = df_tdi[(df_tdi[col] != '--') & (df_tdi[col] != '***')]
        df_tdi[colunas_interesse] = df_tdi[colunas_interesse].astype(float)
        df_tdi = df_tdi.fillna(df_tdi.mean(numeric_only=True))

        return df_tdi

    
    def create_rmd_table(self):
        df_rmd = pd.read_excel('data/Remuneracao_docentes_Municipios_2020.xlsx', skiprows=8, usecols=[2,4,5,6,7,8,9,10,11,12,13,14])

        df_rmd = (
            df_rmd.query("NO_DEPENDENCIA == 'Municipal' and NO_CATEGORIA == 'Total'")
            .drop(columns=['NO_DEPENDENCIA', 'NO_CATEGORIA'])
            .rename(columns={'SG_UF': 'NO_UF'})
        )

        colunas_interesse = ['ED_BAS_CAT1', 'ED_BAS_CAT2', 'ED_BAS_CAT3', 'ED_BAS_CAT4', 'ED_BAS_CAT5', 'ED_BAS_CAT6', 'ED_BAS_CAT7', 'ED_BAS_CAT8']
        
        for col in colunas_interesse:
            df_rmd = df_rmd[(~df_rmd[col].isin(['a', 'd', 'c']))]
        df_rmd[colunas_interesse] = df_rmd[colunas_interesse].astype(float)
        df_rmd = df_rmd.fillna(df_rmd.mean(numeric_only=True))
        return df_rmd   

    def create_tnr_table(self):
        df_tnr = pd.read_excel('data/TNR_MUNICIPIOS_2021.xlsx', skiprows=8, usecols=[2,4,5,6,19])

        df_tnr = (
            df_tnr.query("NO_DEPENDENCIA == 'Pública' and NO_CATEGORIA == 'Total'")
            .drop(columns=['NO_DEPENDENCIA', 'NO_CATEGORIA'])
            .rename(columns={'SG_UF': 'NO_UF', '4_CAT_MED': '4_CAT_MED_tnr'})
        )
        colunas_interesse = ['4_CAT_MED_tnr']
        for col in colunas_interesse:
            df_tnr = df_tnr[(df_tnr[col] != '--') & (df_tnr[col] != '***')]
        df_tnr[colunas_interesse] = df_tnr[colunas_interesse].astype(float)
        df_tnr = df_tnr.fillna(df_tnr.mean(numeric_only=True))

        return df_tnr     


    def create_rendimento_table(self):
        df_rendimento = pd.read_excel('data/tx_rend_municipios_2021.xlsx', skiprows=8, usecols=[2,4,5,6,19,37,55])

        df_rendimento = (
            df_rendimento.query("NO_DEPENDENCIA == 'Pública' and NO_CATEGORIA == 'Total'")
            .drop(columns=['NO_DEPENDENCIA', 'NO_CATEGORIA'])
            .rename(columns={'SG_UF': 'NO_UF', 
                            '1_CAT_MED': 'tx_aprovacao_EM',
                            '2_CAT_MED': 'tx_reprovacao_EM',
                            '3_CAT_MED': 'tx_abandono_EM',})
        )
        colunas_interesse = ['tx_aprovacao_EM', 'tx_reprovacao_EM', 'tx_abandono_EM']
        for col in colunas_interesse:
            df_rendimento = df_rendimento[(df_rendimento[col] != '--') & (df_rendimento[col] != '***')]
        df_rendimento[colunas_interesse] = df_rendimento[colunas_interesse].astype(float)
        df_rendimento = df_rendimento.fillna(df_rendimento.mean(numeric_only=True))

        return df_rendimento



    def combine_data(self, df_transicao, df_inse, df_microdados, df_afd, df_ied, df_ideb, df_atu, df_had, df_dsu, df_ird, df_tdi, df_rmd, df_tnr, df_rendimento):
        df_combined = (df_transicao
                       .merge(df_inse, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_microdados, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_afd, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_ied, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_ideb, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_atu, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_had, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_dsu, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_ird, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_tdi, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_rmd, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_tnr, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .merge(df_rendimento, on=['NO_UF', 'NO_MUNICIPIO'], how='left')
                       .dropna())
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