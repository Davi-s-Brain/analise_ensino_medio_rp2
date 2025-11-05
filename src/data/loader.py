import re
import csv
import os
import numpy as np
import unicodedata
import pandas as pd
from zipfile import ZipFile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
        self.feature_names = None

    def prepare_data(self, df, test_size=0.2, random_state=42):
        # --- 1. Remover Outliers (DECIDIMOS NÃO FAZER) ---
        # df = self._remove_outliers(df) # Comentado
        
        # --- 2. Definir Alvo (y) e Limpar NaNs do Alvo ---
        target = 'tx_evasao_total_EM'
        if target not in df.columns:
            print(f"ERRO CRÍTICO: Coluna alvo '{target}' não encontrada. Verifique o combine_data.")
            return None, None, None, None

        original_len = len(df)
        df = df.dropna(subset=[target])
        print(f"Removidas {original_len - len(df)} linhas por alvo ('{target}') ausente.")
        y = pd.qcut(df[target], q=4, labels=['Baixa Evasão', 'Média Baixa', 'Média Alta', 'Alta Evasão'])

        
        # --- 3. Definir Lista de Features (X) ---
        
        # Lista de features (com o VAZAMENTO DE DADOS REMOVIDO)
        feature_list = [
            'NO_REGIAO', 'NO_UF',
            
            # INSE
            'MEDIA_INSE', 'PC_NIVEL_1', 'PC_NIVEL_2', 'PC_NIVEL_3', 'PC_NIVEL_4', 'PC_NIVEL_5', 'PC_NIVEL_6', 'PC_NIVEL_7',

            # Microdados
            'QT_DOC_MED_mean', 'QT_DOC_MED_std', 'QT_DOC_MED_min', 'QT_DOC_MED_max', 'QT_SALAS_EXISTENTES_mean', 'QT_SALAS_EXISTENTES_std', 'QT_SALAS_EXISTENTES_min', 'QT_SALAS_EXISTENTES_max', 'QT_FUNCIONARIOS_mean', 'QT_FUNCIONARIOS_std', 'QT_FUNCIONARIOS_min', 'QT_FUNCIONARIOS_max', 'IN_PREDIO_COMPARTILHADO_mean', 'IN_AGUA_INEXISTENTE_mean', 'IN_ENERGIA_INEXISTENTE_mean', 'IN_ESGOTO_INEXISTENTE_mean', 'IN_BIBLIOTECA_mean', 'IN_LABORATORIO_INFORMATICA_mean', 'IN_QUADRA_ESPORTES_mean', 'IN_REFEITORIO_mean', 'IN_INTERNET_mean', 'IN_INTERNET_ALUNOS_mean', 'IN_BANDA_LARGA_mean', 'IN_PROF_PSICOLOGO_mean', 'IN_PROF_ASSIST_SOCIAL_mean', 'IN_EXAME_SELECAO_mean', 'IN_ORGAO_GREMIO_ESTUDANTIL_mean', 'IN_FINAL_SEMANA_mean', 'QT_MAT_MED_sum', 'QT_MAT_MED_INT_sum',
            
            # Adequação da formação docente (AFD)
            'MED_CAT_1_afd', 'MED_CAT_2_afd', 'MED_CAT_3_afd', 'MED_CAT_4_afd', 'MED_CAT_5_afd',

            # Indicador de esforço docente (IED)
            'MED_CAT_1_ied', 'MED_CAT_2_ied', 'MED_CAT_3_ied', 'MED_CAT_4_ied', 'MED_CAT_5_ied', 'MED_CAT_6_ied',

            # Índice de Desenvolvimento da Educação Básica (Ideb) (Muletas)
            'VL_NOTA_MATEMATICA_2021', 'VL_NOTA_PORTUGUES_2021', 'VL_NOTA_MEDIA_2021', 'VL_OBSERVADO_2021', 'VL_PROJECAO_2021',

            # Média de alunos por turma (ATU)
            'MED_CAT_0_atu',

            # Média Horas-aula diária (HAD)
            'MED_CAT_0_had',

            # Percentual de docentes com curso superior (DSU)
            'MED_CAT_0_dsu',

            # Regularidade do corpo docente (IRD)
            'EDU_BAS_CAT_1', 'EDU_BAS_CAT_2', 'EDU_BAS_CAT_3', 'EDU_BAS_CAT_4',

            # Taxa de distorção idade série (z) (Muletas)
            'MED_CAT_0_tdi', 'MED_01_CAT_0_tdi', 'MED_02_CAT_0_tdi', 'MED_03_CAT_0_tdi'

            # Remuneração média dos docentes (RMD)
            'ED_BAS_CAT1', 'ED_BAS_CAT2', 'ED_BAS_CAT3', 'ED_BAS_CAT4', 'ED_BAS_CAT5', 'ED_BAS_CAT6', 'ED_BAS_CAT7', 'ED_BAS_CAT8',
            
            
            # Dados IBGE - PIB (Nomes longos)
            # 'IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES', 'PARTICIPACAO_DO_PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES_NO_PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES_DA_GRANDE_REGIAO', 'PARTICIPACAO_DO_PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES_NO_PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES_DA_MESORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES_NO_PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES_DA_MICRORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES_NO_PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES_DA_UNIDADE_DA_FEDERACAO', 'PARTICIPACAO_DO_PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES_NO_PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES_DO_BRASIL', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_DA_GRANDE_REGIAO', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_DA_MESORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_DA_MICRORREGIAO_GEOGRAFIC', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_DA_UNIDADE_DA_FEDERACAO', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_DO_BRASIL', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_DA_GRANDE_REGIAO', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_DA_MESORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_DA_MICRORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_DA_UNIDADE_DA_FEDERACAO', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_DO_BRASIL', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_DA_GRANDE_REGIAO', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_DA_MESORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_DA_MICRORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_DA_UNIDADE_DA_FEDERACAO', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_DO_BRASIL', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_DA_GRANDE_REGIAO', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_DA_MESORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_DA_MICRORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_DA_UNIDADE_DA_FEDERACAO', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_DO_BRASIL', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL_DA_GRANDE_REGIAO', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL_DA_MESORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL_DA_MICRORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL_DA_UNIDADE_DA_FEDERACAO', 'PARTICIPACAO_DO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL_NO_VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL_DO_BRASIL', 'PARTICIPACAO_DOS_IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES_NOS_IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES_DA_GRANDE_REGIAO', 'PARTICIPACAO_DOS_IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES_NOS_IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES_DA_MESORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DOS_IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES_NOS_IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES_DA_MICRORREGIAO_GEOGRAFICA', 'PARTICIPACAO_DOS_IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES_NOS_IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES_DA_UNIDADE_DA_FEDERACAO', 'PARTICIPACAO_DOS_IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES_NOS_IMPOSTOS_LIQUIDOS_DE_SUBSIDIOS_SOBRE_PRODUTOS_A_PRECOS_CORRENTES_DO_BRASIL', 'PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES', 'VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL', 'VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_AGROPECUARIA', 'VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DA_INDUSTRIA', 'VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_DOS_SERVICOS_EXCLUSIVE_ADMINISTRACAO_DEFESA_EDUCACAO_E_SAUDE_PUBLICAS_E_SEGURIDADE_SOCIAL', 'VALOR_ADICIONADO_BRUTO_A_PRECOS_CORRENTES_TOTAL',
            
            # Engenharia de features
            'RISCO_SOCIAL_TDI_PIB', # Muleta
            'RISCO_PEDAGOGICO_TDI_ATU', # Muleta
            'RISCO_INFRA_TDI_NET', # Muleta
            'RISCO_GOVERNANCA_IDH', # Muleta
            
            
            # IDH (Índice de Desenvolvimento Humano)
            'ADH_IDHM',
            'ADH_IDHM_E',
            'ADH_IDHM_L',
            'ADH_IDHM_R',
            'ADH_INDICE_GINI',
            'ADH_PERC_POPULACAO_RURAL',
            'ADH_TX_ANALFABETISMO_25_MAIS',
            'ADH_EXPECTATIVA_ANOS_ESTUDO',
            'ADH_TX_ATRASO_2_FUNDAMENTAL', # Muleta
            'ADH_RENDA_PER_CAPITA',
            'ADH_PROP_POBREZA_EXTREMA',
            'ADH_PROP_VULNER_POBREZA',
            
            # Dados IBGE - Raça e gênero
            'CENSO_PERC_HOMENS', 'CENSO_PERC_MULHERES', 'RACA_PERC_INDIGENA', 'RACA_PERC_PRETA_PARDA',
            
            # Dados Bolsa Família
            'BF_QTD_FAMILIAS_MEDIA_MENSAL',
            'BF_VALOR_ANUAL_TOTAL',
            'BF_VALOR_MEDIO_POR_FAMILIA_ANUAL',
            'BF_PERC_POPULACAO'
        ]
        
        # Filtra a lista para apenas colunas que REALMENTE existem no df
        cols_existentes = [col for col in feature_list if col in df.columns]
        cols_faltantes = [col for col in feature_list if col not in df.columns]
        
        if cols_faltantes:
            print(f"Aviso: {len(cols_faltantes)} colunas da lista não foram encontradas no DataFrame e serão ignoradas.")

        X = df[cols_existentes].copy()
        
        # --- 4. Encoding de Categóricas ---
        # dummy_na=True cria uma coluna "NO_REGIAO_nan", o que pode ser útil
        print("Aplicando One-Hot Encoding em colunas categóricas...")
        X = pd.get_dummies(X, columns=['NO_REGIAO', 'NO_UF'], dummy_na=True)
        
        # Salva os nomes *depois* do get_dummies
        self.feature_names = X.columns.tolist()
        
        
        # --- 5. Divisão Treino/Teste ---
        print("Dividindo dados em treino (80%) e teste (20%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        
        # --- 6. Imputação (Preenchimento de NaNs) ---
        print("Preenchendo valores ausentes (NaN) com a mediana do treino...")
        imputer = SimpleImputer(strategy='median')
        
        # "Fita" (treina) o imputer APENAS com X_train
        imputer.fit(X_train)
        
        # "Transforma" (preenche) ambos os conjuntos
        X_train_imputed_data = imputer.transform(X_train)
        X_test_imputed_data = imputer.transform(X_test)

        # Converte de volta para DataFrame para manter nomes de colunas e índices
        X_train_imputed = pd.DataFrame(
            X_train_imputed_data, columns=self.feature_names, index=X_train.index
        )
        X_test_imputed = pd.DataFrame(
            X_test_imputed_data, columns=self.feature_names, index=X_test.index
        )
        
        
        # --- 7. Scaling (Normalização) ---
        print("Escalando (normalizando) os dados...")
        
        # "Fita" (treina) o scaler APENAS com X_train_imputed
        self.scaler.fit(X_train_imputed)
        
        # "Transforma" (escala) ambos
        X_train_scaled_data = self.scaler.transform(X_train_imputed)
        X_test_scaled_data = self.scaler.transform(X_test_imputed)

        # Converte de volta para DataFrame (bom para depuração e feature importance)
        X_train_scaled = pd.DataFrame(
            X_train_scaled_data, columns=self.feature_names, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled_data, columns=self.feature_names, index=X_test.index
        )

        print("\nProcessamento de dados concluído.")
        
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
        df_tnr = pd.read_excel('data/tnr_municipios_2021.xlsx', skiprows=8, usecols=[2,4,5,6,19])

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


    def create_pib_table_ibge(self, full_path_in, full_path_out):
        """
        Versão corrigida que lê o arquivo CSV completo do IBGE,
        ignorando os múltiplos rodapés e capturando todas as variáveis.
        """

        data_rows = []
        current_variable_name = None
        variaveis_encontradas = set()

        # Primeiro, verifica se o arquivo limpo já existe
        if os.path.exists(full_path_out):
            print(f"Arquivo PIB limpo já existe ('{full_path_out}'). Carregando diretamente.")
            df_final = pd.read_csv(full_path_out)
            
            # Verificação de segurança para o erro que vimos
            if 'CO_MUNICIPIO_7' not in df_final.columns:
                print("--- ATENÇÃO ---")
                print("O arquivo cacheado NÃO tem 'CO_MUNICIPIO_7'.")
                print(f"Apague o arquivo '{full_path_out}' e rode novamente para recriá-lo.")
                print("---------------")

            return df_final
        else:
            print(f"Arquivo PIB limpo não encontrado. Processando '{full_path_in}'...")
            try:
                with open(full_path_in, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter=';')
                    
                    for i, row in enumerate(reader):
                        if not row:
                            continue  # Pula linhas em branco
                        
                        first_cell = row[0].strip()
                        
                        if first_cell.startswith("Variável - "):
                            var_name = first_cell.replace("Variável - ", "").strip()
                            var_name = re.sub(r'\s\([\w\s%]+\)$', '', var_name).strip()
                            
                            if var_name and "Nível" not in var_name:
                                current_variable_name = var_name
                                if var_name not in variaveis_encontradas:
                                    variaveis_encontradas.add(var_name)
                        
                        elif first_cell == "MU":
                            if current_variable_name and len(row) >= 4:
                                cod_mun = row[1]
                                nome_mun_raw = row[2]
                                valor = row[3]
                                
                                data_rows.append([cod_mun, nome_mun_raw, current_variable_name, valor])
                        
                        elif first_cell.startswith("Fonte:") or \
                            first_cell == "Legenda" or \
                            first_cell == "Notas" or \
                            first_cell == "Nível" or \
                            first_cell.startswith("Símbolo") or \
                            first_cell.startswith("\"Tabela"):
                            pass 

                if not data_rows:
                    print("Nenhum dado de município foi encontrado. Verifique o arquivo.")
                    return pd.DataFrame() # Retorna DF vazio

                # --- Transformação e Pivotagem ---
                df_long = pd.DataFrame(
                    data_rows, 
                    # <--- CORREÇÃO 2: Renomeia a chave para o nome correto ---
                    columns=['CO_MUNICIPIO_7', 'NO_MUNICIPIO_RAW', 'Variavel', 'Valor']
                )
                
                df_long = df_long.drop_duplicates()
                
                df_long_agg = df_long.groupby(['CO_MUNICIPIO_7', 'NO_MUNICIPIO_RAW', 'Variavel']).first().reset_index()

                df_pivot = df_long_agg.pivot_table(
                    # <--- CORREÇÃO 2 (continuação) ---
                    index=['CO_MUNICIPIO_7', 'NO_MUNICIPIO_RAW'],
                    columns='Variavel',
                    values='Valor',
                    aggfunc='first'
                ).reset_index()
                
                df_pivot.columns.name = None

                # --- Limpeza Final ---
                df_pivot['NO_MUNICIPIO'] = df_pivot['NO_MUNICIPIO_RAW'].str.replace(r'\s\([A-Z]{2}\)$', '', regex=True)
                # <--- CORREÇÃO 2 (continuação) ---
                df_pivot['CO_MUNICIPIO_7'] = df_pivot['CO_MUNICIPIO_7'].astype(int)
                
                colunas_indicadores = [
                    col for col in df_pivot.columns 
                    # <--- CORREÇÃO 2 (continuação) ---
                    if col not in ['CO_MUNICIPIO_7', 'NO_MUNICIPIO_RAW', 'NO_MUNICIPIO']
                ]
                
                for col in colunas_indicadores:
                    val_str = df_pivot[col].astype(str)
                    val_str = val_str.str.replace(r'\.', '', regex=False)
                    val_str = val_str.str.replace(r',', '.', regex=False)
                    df_pivot[col] = pd.to_numeric(val_str, errors='coerce')

                # <--- CORREÇÃO 3: Mantém a coluna CO_MUNICIPIO_7 no df_final ---
                colunas_finais = ['CO_MUNICIPIO_7', 'NO_MUNICIPIO'] + colunas_indicadores
                df_final = df_pivot[colunas_finais].copy() # Usar .copy() para evitar warnings
                
                df_final = df_final.dropna(subset=colunas_indicadores, how='all')
                
                # --- Padronização de Nomes de Colunas ---
                novos_nomes = []
                for col in df_final.columns:
                    col = str(col)
                    col_nfkd = unicodedata.normalize('NFKD', col)
                    col_ascii = col_nfkd.encode('ASCII', 'ignore').decode('utf-8', 'ignore')
                    col_upper = col_ascii.upper()
                    col_cleaned = re.sub(r'[^A-Z0-9_]+', '_', col_upper)
                    col_cleaned = re.sub(r'__+', '_', col_cleaned)
                    col_cleaned = col_cleaned.strip('_')
                    novos_nomes.append(col_cleaned)
                
                df_final.columns = novos_nomes
                # Verificação final: Garante que a coluna-chave (agora em maiúsculo) existe
                if 'CO_MUNICIPIO_7' not in df_final.columns:
                    print("ERRO PÓS-RENOMEAÇÃO: 'CO_MUNICIPIO_7' não encontrada.")
                
                print(f"Colunas indicadores (features do PIB): {len(colunas_indicadores)}")
                
                print("\n--- Amostra dos Dados Finais do PIB (Head) ---")
                print(df_final.head())
                
                print(f"Salvando PIB limpo e processado em: {full_path_out}")
                df_final.to_csv(full_path_out, index=False, encoding='utf-8-sig')
                
                return df_final

            except FileNotFoundError:
                print(f"Erro: O arquivo '{full_path_in}' não foi encontrado.")
                return pd.DataFrame() # Retorna DF vazio
            except Exception as e:
                print(f"Ocorreu um erro inesperado ao processar o PIB: {e}")
                return pd.DataFrame() # Retorna DF vazio


    def combine_data(self, df_transicao, df_inse, df_microdados, df_afd, df_ied, df_ideb, df_atu, df_had, df_dsu, df_ird, df_tdi, df_rmd, df_tnr, df_rendimento, df_ibge, df_idh, df_raca_genero, df_bolsa_familia):
        if 'ADH_NO_MUNICIPIO' in df_idh.columns:
            df_idh = df_idh.drop(columns=['ADH_NO_MUNICIPIO'])
            print("Coluna 'ADH_NO_MUNICIPIO' removida do df_idh.")
            
        if 'CENSO_NO_MUNICIPIO' in df_raca_genero.columns:
            df_raca_genero = df_raca_genero.drop(columns=['CENSO_NO_MUNICIPIO'])
            print("Coluna 'CENSO_NO_MUNICIPIO' removida do df_raca_genero.")

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
                       .merge(df_ibge, on=['NO_MUNICIPIO'], how='left')
                       .merge(df_idh, on=['CO_MUNICIPIO_7'], how='left')
                       .merge(df_raca_genero, on=['CO_MUNICIPIO_7'], how='left'))
        
        df_combined['CO_MUNICIPIO_6'] = df_combined['CO_MUNICIPIO_7'] // 10
        
        df_combined = df_combined.merge(df_bolsa_familia, on=['CO_MUNICIPIO_6'], how='left')
        
        print("Iniciando Engenharia de Features (criando colunas de Risco)...")
        
        if 'BF_QTD_FAMILIAS_MEDIA_MENSAL' in df_combined.columns and 'ADH_POPULACAO_TOTAL' in df_combined.columns:
            print("Criando feature 'BF_PERC_POPULACAO'...")
            # Assumindo 2.5 pessoas por família (média nacional)
            populacao_beneficiaria = df_combined['BF_QTD_FAMILIAS_MEDIA_MENSAL'] * 2.5 
            pop_total_safe = df_combined['ADH_POPULACAO_TOTAL'].replace(0, np.nan)

            # Calcula a % da população (de 2010) que recebe benefício (em 2021)
            df_combined['BF_PERC_POPULACAO'] = (populacao_beneficiaria / pop_total_safe)
            df_combined['BF_PERC_POPULACAO'] = df_combined['BF_PERC_POPULACAO'].fillna(0)
        
        # Assumindo que a coluna do PIB já está limpa e é numérica
        pib_col = 'PRODUTO_INTERNO_BRUTO_A_PRECOS_CORRENTES'
        
        # Garante que a coluna existe e é numérica
        if pib_col in df_combined.columns:
            df_combined[pib_col] = pd.to_numeric(df_combined[pib_col], errors='coerce').fillna(1)
            df_combined[pib_col] = df_combined[pib_col].replace(0, 1) # Evita divisão por zero
            
            # Feature 1: Risco Social (Distorção é pior em cidades mais pobres?)
            df_combined['RISCO_SOCIAL_TDI_PIB'] = df_combined['MED_CAT_0_tdi'] / df_combined[pib_col]
        
        # Feature 2: Risco Pedagógico (Distorção é pior em turmas cheias?)
        if 'MED_CAT_0_atu' in df_combined.columns:
            df_combined['RISCO_PEDAGOGICO_TDI_ATU'] = df_combined['MED_CAT_0_tdi'] * df_combined['MED_CAT_0_atu']

        # Feature 3: Risco Infra (Distorção é pior sem internet?)
        if 'IN_BANDA_LARGA_mean' in df_combined.columns:
            df_combined['RISCO_INFRA_TDI_NET'] = df_combined['MED_CAT_0_tdi'] * (1 - df_combined['IN_BANDA_LARGA_mean'])
            
        # Feature 4: Risco Governança (Distorção é pior sem grêmio estudantil e baixo IDH-E?)
        df_combined['RISCO_GOVERNANCA_IDH'] = (1 - df_combined['IN_ORGAO_GREMIO_ESTUDANTIL_mean']) * (1 - df_combined['ADH_IDHM_E'])
        
        print("Engenharia de Features concluída.")
        df_combined.to_csv('data/data_combined.csv', index=False, encoding='utf-8-sig')
        
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
    
    def create_idh_table(self, file_path='data/mundo_onu_adh.csv'):
        """
        Carrega e processa os dados do Atlas de Desenvolvimento Humano (ADH).
        Filtra pelas colunas mais relevantes e renomeia-as para o merge.
        (Arquivo baixado do Atlas Brasil / Base dos Dados - Censo 2010)
        """
        try:
            # Carregar o CSV
            df_adh = pd.read_csv(file_path, encoding='latin1')
        except FileNotFoundError:
            print(f"Erro: Arquivo '{file_path}' não foi encontrado.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Erro ao ler o arquivo: {e}")
            return pd.DataFrame()

        # --- 1. Seleção das Colunas de Interesse ---
        colunas_interesse = [
            'id_municipio', 'id_municipio_nome',
            'idhm', 'idhm_e', 'idhm_l', 'idhm_r',
            'indice_gini',
            'populacao_rural', 'populacao_urbana', 'populacao',
            'taxa_analfabetismo_25_mais', 'expectativa_anos_estudo', 
            'taxa_atraso_2_fundamental',
            'renda_pc', 'prop_pobreza_extrema', 'prop_vulner_pobreza'
        ]
        
        colunas_existentes = [col for col in colunas_interesse if col in df_adh.columns]
        df_filtrado = df_adh[colunas_existentes].copy()

        # --- 2. Engenharia de Features (Percentual Rural) ---
        if 'populacao' in df_filtrado.columns and 'populacao_rural' in df_filtrado.columns:
            pop_total_safe = df_filtrado['populacao'].replace(0, np.nan)
            df_filtrado['perc_populacao_rural'] = (df_filtrado['populacao_rural'] / pop_total_safe) * 100
            df_filtrado['perc_populacao_rural'] = df_filtrado['perc_populacao_rural'].fillna(0)

        # --- 3. Renomeação e Padronização (Padrão ADH_) ---
        rename_map = {
            'id_municipio': 'CO_MUNICIPIO_7', # Chave de merge!
            'id_municipio_nome': 'ADH_NO_MUNICIPIO',
            'idhm': 'ADH_IDHM',
            'idhm_e': 'ADH_IDHM_E',
            'idhm_l': 'ADH_IDHM_L',
            'idhm_r': 'ADH_IDHM_R',
            'indice_gini': 'ADH_INDICE_GINI',
            'populacao_rural': 'ADH_POPULACAO_RURAL',
            'populacao_urbana': 'ADH_POPULACAO_URBANA',
            'populacao': 'ADH_POPULACAO_TOTAL',
            'taxa_analfabetismo_25_mais': 'ADH_TX_ANALFABETISMO_25_MAIS',
            'expectativa_anos_estudo': 'ADH_EXPECTATIVA_ANOS_ESTUDO',
            'taxa_atraso_2_fundamental': 'ADH_TX_ATRASO_2_FUNDAMENTAL',
            'renda_pc': 'ADH_RENDA_PER_CAPITA',
            'prop_pobreza_extrema': 'ADH_PROP_POBREZA_EXTREMA',
            'prop_vulner_pobreza': 'ADH_PROP_VULNER_POBREZA',
            'perc_populacao_rural': 'ADH_PERC_POPULACAO_RURAL'
        }
        
        df_final = df_filtrado.rename(columns={k: v for k, v in rename_map.items() if k in df_filtrado.columns})
        
        colunas_finais = [col for col in rename_map.values() if col in df_final.columns]
        df_pronto = df_final[colunas_finais].copy()

        print("Dados do ADH prontos para o merge.")
        return df_pronto
    

    def create_raca_table(self, zip_file_path='data/POP_COR_SEXO.zip', cached_file_path='data/RACA_MUNICIPIOS_LIMPADO.csv'):
        """
        Versão FINAL (v10) - "Turbinada"
        Lê o arquivo 'longo' (formato CSV, separado por ';') que está dentro do ZIP.
        Processa:
        1. Nomes dos Municípios
        2. Dados de Raça/Cor
        3. Dados de Sexo
        E salva tudo em um único arquivo de cache.
        """
        
        # --- 1. Verificação de Cache ---
        if os.path.exists(cached_file_path):
            print(f"Arquivo cacheado de Censo Completo '{cached_file_path}' encontrado. Carregando diretamente.")
            try:
                df_final = pd.read_csv(cached_file_path)
                return df_final
            except Exception as e:
                print(f"Erro ao ler o cache '{cached_file_path}'. Recriando... Erro: {e}")
                
        print(f"Cache '{cached_file_path}' não encontrado. Iniciando processamento completo de '{zip_file_path}'...")
        
        temp_dir = 'data/temp_sidra_censo'
        extracted_file_path = None
        
        try:
            # --- 2. Extrair o ZIP ---
            with ZipFile(zip_file_path, 'r') as zip_ref:
                extracted_files = zip_ref.namelist()
                csv_file_name = [f for f in extracted_files if not f.startswith('__MACOSX')][0]
                os.makedirs(temp_dir, exist_ok=True)
                zip_ref.extract(csv_file_name, path=temp_dir)
                extracted_file_path = os.path.join(temp_dir, csv_file_name)

            # --- 3. Carregar o CSV extraído ---
            col_names = [
                'ANO', 'MUNICIPIO_COD_RAW', 'MUNICIPIO_NOME', 'VARIAVEL',
                'SITUACAO_DOMICILIO', 'SEXO', 'COR_RACA', 'FAIXA_ETARIA', 'VALOR'
            ]
            print(f"Carregando arquivo {csv_file_name} (pode demorar)...")
            df = pd.read_csv(
                extracted_file_path, 
                delimiter=';', skiprows=2, header=None, names=col_names,
                encoding='utf-8', low_memory=False,
                converters={'VALOR': lambda x: pd.to_numeric(str(x).replace(',', '.'), errors='coerce')}
            )
            print("Arquivo bruto carregado.")

            # --- 4. Filtro Mestre ---
            # Filtra apenas a variável que queremos (População) e a Situação (Total)
            var_correta = 'População residente (Pessoas)' 
            df_filtrado_mestre = df[
                (df['VARIAVEL'] == var_correta) &
                (df['SITUACAO_DOMICILIO'] == 'Total')
            ].copy()
            
            if df_filtrado_mestre.empty:
                print(f"ERRO: Variável '{var_correta}' não encontrada.")
                return pd.DataFrame()

            # --- 5. Bloco 1: Processar RAÇA ---
            print("Processando dados de Raça/Cor...")
            df_raca_filtrado = df_filtrado_mestre[
                (df_filtrado_mestre['SEXO'] == 'Total') &
                (df_filtrado_mestre['COR_RACA'] != 'Total')
            ].copy()
            df_raca_agrupado = df_raca_filtrado.groupby(
                ['MUNICIPIO_COD_RAW', 'COR_RACA']
            )['VALOR'].sum().reset_index()
            df_raca_agrupado['COR_RACA'] = df_raca_agrupado['COR_RACA'].replace({
                'Indígena': 'Indigena', 'Sem declaração': 'Sem_Declaracao'
            })
            df_raca_pivot = df_raca_agrupado.pivot_table(
                index='MUNICIPIO_COD_RAW', columns='COR_RACA', values='VALOR', aggfunc='first'
            ).reset_index()
            df_raca_pivot['CO_MUNICIPIO_7'] = pd.to_numeric(df_raca_pivot['MUNICIPIO_COD_RAW'].str.split(' ').str[0], errors='coerce').astype(int)
            pop_cols = ['Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
            for col in pop_cols:
                if col not in df_raca_pivot.columns: df_raca_pivot[col] = 0
            df_raca_pivot[pop_cols] = df_raca_pivot[pop_cols].fillna(0)
            df_raca_pivot['RACA_POP_TOTAL'] = df_raca_pivot[pop_cols].sum(axis=1)
            df_raca_pivot['RACA_POP_TOTAL'] = df_raca_pivot['RACA_POP_TOTAL'].replace(0, np.nan)
            df_raca_pivot['RACA_PERC_PRETA_PARDA'] = (df_raca_pivot['Preta'] + df_raca_pivot['Parda']) / df_raca_pivot['RACA_POP_TOTAL']
            df_raca_pivot['RACA_PERC_INDIGENA'] = df_raca_pivot['Indigena'] / df_raca_pivot['RACA_POP_TOTAL']
            df_raca_final = df_raca_pivot[['CO_MUNICIPIO_7', 'RACA_PERC_PRETA_PARDA', 'RACA_PERC_INDIGENA']].fillna(0)

            # --- 6. Bloco 2: Processar SEXO ---
            print("Processando dados de Sexo...")
            df_sexo_filtrado = df_filtrado_mestre[
                (df_filtrado_mestre['COR_RACA'] == 'Total') &
                (df_filtrado_mestre['SEXO'] != 'Total')
            ].copy()
            df_sexo_agrupado = df_sexo_filtrado.groupby(
                ['MUNICIPIO_COD_RAW', 'SEXO']
            )['VALOR'].sum().reset_index()
            df_sexo_pivot = df_sexo_agrupado.pivot_table(
                index='MUNICIPIO_COD_RAW', columns='SEXO', values='VALOR', aggfunc='first'
            ).reset_index()
            df_sexo_pivot['CO_MUNICIPIO_7'] = pd.to_numeric(df_sexo_pivot['MUNICIPIO_COD_RAW'].str.split(' ').str[0], errors='coerce').astype(int)
            pop_cols_sexo = ['Homens', 'Mulheres']
            for col in pop_cols_sexo:
                if col not in df_sexo_pivot.columns: df_sexo_pivot[col] = 0
            df_sexo_pivot[pop_cols_sexo] = df_sexo_pivot[pop_cols_sexo].fillna(0)
            df_sexo_pivot['SEXO_POP_TOTAL'] = df_sexo_pivot[pop_cols_sexo].sum(axis=1)
            df_sexo_pivot['SEXO_POP_TOTAL'] = df_sexo_pivot['SEXO_POP_TOTAL'].replace(0, np.nan)
            df_sexo_pivot['CENSO_PERC_HOMENS'] = df_sexo_pivot['Homens'] / df_sexo_pivot['SEXO_POP_TOTAL']
            df_sexo_pivot['CENSO_PERC_MULHERES'] = df_sexo_pivot['Mulheres'] / df_sexo_pivot['SEXO_POP_TOTAL']
            df_sexo_final = df_sexo_pivot[['CO_MUNICIPIO_7', 'CENSO_PERC_HOMENS', 'CENSO_PERC_MULHERES']].fillna(0)

            # --- 7. Bloco 3: Processar NOMES ---
            print("Processando Nomes de Municípios...")
            df_nomes = df_filtrado_mestre[['MUNICIPIO_COD_RAW', 'MUNICIPIO_NOME']].drop_duplicates().copy()
            df_nomes['CO_MUNICIPIO_7'] = pd.to_numeric(df_nomes['MUNICIPIO_COD_RAW'].str.split(' ').str[0], errors='coerce').astype(int)
            # Limpa o nome (ex: "Alta Floresta D'Oeste (RO)")
            df_nomes['CENSO_NO_MUNICIPIO'] = df_nomes['MUNICIPIO_NOME'].str.replace(r'\s\([A-Z]{2}\)$', '', regex=True)
            df_nomes_final = df_nomes[['CO_MUNICIPIO_7', 'CENSO_NO_MUNICIPIO']].dropna().drop_duplicates()

            # --- 8. Merge Final ---
            print("Juntando tabelas de Raça, Sexo e Nomes...")
            df_final = df_nomes_final.merge(df_raca_final, on='CO_MUNICIPIO_7', how='left')
            df_final = df_final.merge(df_sexo_final, on='CO_MUNICIPIO_7', how='left')
            
            # --- 9. Salvar no Cache ---
            print(f"Salvando dados processados no cache: '{cached_file_path}'")
            df_final.to_csv(cached_file_path, index=False)
            
            print("Dados do Censo (Raça, Sexo, Nomes) processados e prontos para o merge.")
            return df_final

        except Exception as e:
            print(f"Erro ao processar o arquivo de Censo: {e}")
            return pd.DataFrame()
        finally:
            # --- 10. Limpeza do Temp ---
            if extracted_file_path and os.path.exists(extracted_file_path):
                try:
                    os.remove(extracted_file_path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except Exception as e:
                    print(f"Aviso: Não foi possível limpar o arquivo temporário. Erro: {e}")
                    
    
    def create_bolsa_familia_table(self, file_path='data/bolsa_familia_2021.csv', cached_file_path='data/BOLSA_FAMILIA_LIMPADO.csv'):
        """
        Carrega os dados do Bolsa Família de 2021.
        Agrega os 12 meses para ter uma média anual por município.
        """
        
        # --- 1. Verificação de Cache ---
        if os.path.exists(cached_file_path):
            print(f"Arquivo cacheado do Bolsa Família '{cached_file_path}' encontrado. Carregando diretamente.")
            try:
                df_final = pd.read_csv(cached_file_path)
                return df_final
            except Exception as e:
                print(f"Erro ao ler o cache '{cached_file_path}'. Recriando... Erro: {e}")
                
        print(f"Cache '{cached_file_path}' não encontrado. Iniciando processamento de '{file_path}'...")
        
        try:
            # --- 2. Carregar o CSV ---
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # --- 3. Limpar Dados ---
            # Renomear colunas
            df = df.rename(columns={
                'ibge': 'CO_MUNICIPIO_6',
                'qtd_familias_beneficiarias_bolsa_familia': 'QTD_FAMILIAS',
                'valor_repassado_bolsa_familia': 'VALOR_REPASSADO'
            })
            
            # Converter colunas para numérico (trata valores faltantes `,,`)
            df['QTD_FAMILIAS'] = pd.to_numeric(df['QTD_FAMILIAS'], errors='coerce').fillna(0)
            df['VALOR_REPASSADO'] = pd.to_numeric(df['VALOR_REPASSADO'], errors='coerce').fillna(0)
            
            # --- 4. Agregação ---
            # Agrupa por município (que é o CO_MUNICIPIO_6)
            print("Agregando dados mensais do Bolsa Família por município...")
            df_agregado = df.groupby('CO_MUNICIPIO_6').agg(
                # Queremos a média de famílias ao longo do ano
                BF_QTD_FAMILIAS_MEDIA_MENSAL=('QTD_FAMILIAS', 'mean'),
                # Queremos o valor total pago no ano
                BF_VALOR_ANUAL_TOTAL=('VALOR_REPASSADO', 'sum')
            ).reset_index()
            
            # --- 5. Engenharia de Feature (Valor por Família) ---
            # Calcula o valor médio anual por família beneficiária
            df_agregado['BF_QTD_FAMILIAS_MEDIA_MENSAL_SAFE'] = df_agregado['BF_QTD_FAMILIAS_MEDIA_MENSAL'].replace(0, np.nan)
            df_agregado['BF_VALOR_MEDIO_POR_FAMILIA_ANUAL'] = df_agregado['BF_VALOR_ANUAL_TOTAL'] / df_agregado['BF_QTD_FAMILIAS_MEDIA_MENSAL_SAFE']
            df_agregado = df_agregado.fillna(0) # Preenche NaNs (onde qtd_familias era 0)

            # --- 6. Seleção Final ---
            colunas_finais = [
                'CO_MUNICIPIO_6',
                'BF_QTD_FAMILIAS_MEDIA_MENSAL',
                'BF_VALOR_ANUAL_TOTAL',
                'BF_VALOR_MEDIO_POR_FAMILIA_ANUAL'
            ]
            df_final = df_agregado[colunas_finais].copy()
            
            # --- 7. Salvar no Cache ---
            print(f"Salvando dados processados do Bolsa Família no cache: '{cached_file_path}'")
            df_final.to_csv(cached_file_path, index=False)
            
            print("Dados do Bolsa Família processados e prontos para o merge.")
            return df_final

        except FileNotFoundError:
            print(f"Erro: Arquivo '{file_path}' não foi encontrado.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Erro ao processar o arquivo do Bolsa Família: {e}")
            return pd.DataFrame()