import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.data.loader import DataLoader

def analyze_data():
    """
    Análise detalhada dos dados para identificar problemas e oportunidades de melhoria
    """
    
    # Carrega os dados
    data_loader = DataLoader('data/TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx')
    
    if os.path.exists('data/data_combined.csv'):
        print("Carregando dados combinados...")
        df = pd.read_csv('data/data_combined.csv', encoding='utf-8-sig')
    else:
        print("Arquivo combinado não encontrado. Execute main.py primeiro.")
        return
    
    print(f"Shape dos dados: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")
    
    # 1. ANÁLISE DA VARIÁVEL TARGET
    print("\n" + "="*50)
    print("1. ANÁLISE DA VARIÁVEL TARGET")
    print("="*50)
    
    target = 'tx_evasao_total_EM'
    if target in df.columns:
        print(f"\nEstatísticas descritivas de {target}:")
        print(df[target].describe())
        
        print(f"\nValores ausentes em {target}: {df[target].isnull().sum()}")
        print(f"Zeros em {target}: {(df[target] == 0).sum()}")
        
        # Distribuição
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(df[target].dropna(), bins=50, alpha=0.7)
        plt.title('Distribuição da Taxa de Evasão')
        plt.xlabel(target)
        
        plt.subplot(1, 3, 2)
        stats.probplot(df[target].dropna(), dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normalidade)')
        
        plt.subplot(1, 3, 3)
        plt.boxplot(df[target].dropna())
        plt.title('Boxplot (Outliers)')
        
        plt.tight_layout()
        plt.savefig('analise_target.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Outliers
        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[target] < lower_bound) | (df[target] > upper_bound)]
        print(f"\nOutliers identificados: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
        
    # 2. ANÁLISE DE CORRELAÇÕES
    print("\n" + "="*50)
    print("2. ANÁLISE DE CORRELAÇÕES")
    print("="*50)
    
    # Seleciona apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols]
    
    print(f"Colunas numéricas encontradas: {len(numeric_cols)}")
    
    # Matriz de correlação
    corr_matrix = df_numeric.corr()
    
    # Correlação com o target
    if target in corr_matrix.columns:
        target_corr = corr_matrix[target].abs().sort_values(ascending=False)
        print(f"\n15 variáveis mais correlacionadas com {target}:")
        print(target_corr.head(15))
        
        # Visualizar top correlações
        plt.figure(figsize=(10, 8))
        top_corr = target_corr.head(20)
        plt.barh(range(len(top_corr)), top_corr.values)
        plt.yticks(range(len(top_corr)), top_corr.index)
        plt.xlabel('Correlação Absoluta')
        plt.title(f'Top 20 Correlações com {target}')
        plt.tight_layout()
        plt.savefig('correlacoes_target.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. ANÁLISE DE MULTICOLINEARIDADE
    print("\n" + "="*50)
    print("3. ANÁLISE DE MULTICOLINEARIDADE")
    print("="*50)
    
    # Encontra pares de variáveis altamente correlacionadas
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.8:  # Correlação alta
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        print(f"Pares com correlação > 0.8 encontrados: {len(high_corr_pairs)}")
        for var1, var2, corr_val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]:
            print(f"{var1} <-> {var2}: {corr_val:.3f}")
    else:
        print("Nenhum par com correlação > 0.8 encontrado")
    
    # 4. ANÁLISE DE MISSING VALUES
    print("\n" + "="*50)
    print("4. ANÁLISE DE MISSING VALUES")
    print("="*50)
    
    missing_stats = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_stats / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_stats,
        'Missing_Percent': missing_percent
    })
    
    # Apenas variáveis com missing values
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_df) > 0:
        print(f"Variáveis com valores ausentes: {len(missing_df)}")
        print("\nTop 15 variáveis com mais missing values:")
        print(missing_df.head(15))
        
        # Visualizar
        if len(missing_df) > 0:
            plt.figure(figsize=(12, 8))
            top_missing = missing_df.head(20)
            plt.barh(range(len(top_missing)), top_missing['Missing_Percent'])
            plt.yticks(range(len(top_missing)), top_missing.index)
            plt.xlabel('Percentual de Missing Values')
            plt.title('Top 20 Variáveis com Missing Values')
            plt.tight_layout()
            plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
            plt.show()
    else:
        print("Nenhuma variável com missing values encontrada")
    
    # 5. ANÁLISE DE VARIÂNCIA
    print("\n" + "="*50)
    print("5. ANÁLISE DE VARIÂNCIA")
    print("="*50)
    
    # Variáveis com baixa variância (potencialmente inúteis)
    variance_stats = df_numeric.var().sort_values()
    low_variance = variance_stats[variance_stats < 0.01]
    
    if len(low_variance) > 0:
        print(f"Variáveis com baixa variância (< 0.01): {len(low_variance)}")
        print(low_variance.head(10))
    else:
        print("Nenhuma variável com variância extremamente baixa encontrada")
    
    # 6. SUMÁRIO E RECOMENDAÇÕES
    print("\n" + "="*50)
    print("6. SUMÁRIO E RECOMENDAÇÕES")
    print("="*50)
    
    print(f"📊 Total de variáveis: {len(df.columns)}")
    print(f"📊 Variáveis numéricas: {len(numeric_cols)}")
    print(f"📊 Observações: {len(df)}")
    
    if target in df.columns:
        current_r2_mlp = 0.42  # Baseado na execução anterior
        current_r2_rf = 0.44
        print(f"📊 R² atual - MLP: {current_r2_mlp:.3f}")
        print(f"📊 R² atual - Random Forest: {current_r2_rf:.3f}")
    
    print("\n🔍 RECOMENDAÇÕES PRINCIPAIS:")
    print("1. Remover ou tratar variáveis com alta correlação (multicolinearidade)")
    print("2. Aplicar feature selection baseada na importância do Random Forest")
    print("3. Considerar transformações na variável target se não estiver normal")
    print("4. Implementar estratégias mais sofisticadas para outliers")
    print("5. Considerar feature engineering (ratios, interactions)")
    print("6. Otimizar hiperparâmetros dos modelos")

if __name__ == "__main__":
    import os
    analyze_data()