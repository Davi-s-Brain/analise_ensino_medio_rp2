import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_transicao_table():
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
      .drop(columns=['NO_DEPENDENCIA'])
    )

    colunas_interesse = ['tx_promocao_EM', 'tx_repetencia_EM', 'tx_evasao_total_EM', 'tx_evasao_1_ano_EM', 'tx_evasao_2_ano_EM', 'tx_evasao_3_ano_EM', 'tx_migracao_eja_EM']

    for col in colunas_interesse:
        df_transicao = df_transicao[(df_transicao[col] != '--') & (df_transicao[col] != '***')]

    df_transicao[colunas_interesse] = df_transicao[colunas_interesse].astype(float)
    df_transicao = df_transicao.dropna(subset=['tx_evasao_total_EM'])
    df_transicao = df_transicao.fillna(df_transicao.mean(numeric_only=True))

    return analyze_data(df_transicao)

      
def analyze_data(df_transicao):
    print("Iniciando analise dos dados de evasão...")
    print("---"*30)
    
    media = df_transicao['tx_evasao_total_EM'].mean()
    mediana = df_transicao['tx_evasao_total_EM'].median()
    desvio_padrao = df_transicao['tx_evasao_total_EM'].std()
    
    # Faça o melhor gráfico possível para a media, mediana e desvio padrão
    print(f"Taxa média de evasão do Ensino Médio: {media:.2f}%") # 9.85%
    print(f"Mediana da taxa de evasão do Ensino Médio: {mediana:.2f}%") # 9.00%
    print(f"Desvio padrão da taxa de evasão do Ensino Médio: {desvio_padrao:.2f}%") # 5.26%
    

    taxas_de_evasao_por_estado = df_transicao.groupby('NO_UF')['tx_evasao_total_EM'].mean().sort_values(ascending=False)
    sns.barplot(x=taxas_de_evasao_por_estado.index, y=taxas_de_evasao_por_estado.values)
    plt.title('Taxa média de evasão do Ensino médio por estado (2021-2022)')
    plt.xlabel('Estado')
    plt.ylabel('Taxa média de evasão (%)')
    # plt.show()
    
    box_plot = sns.boxplot(x=df_transicao['tx_evasao_total_EM'])
    box_plot.set_title('Boxplot da Taxa de Evasão do Ensino Médio (2021-2022)')
    box_plot.set_xlabel('Taxa de Evasão (%)')
    # plt.show()
  

if __name__ == "__main__":
    create_transicao_table()
