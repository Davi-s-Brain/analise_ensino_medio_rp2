import os
import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.models.mlp_model import MLPModel
from src.models.random_forest_model import RandomForestModel
from src.visualization.plots import ModelVisualizer

class CleanDataLoader(DataLoader):
    """
    DataLoader corrigido SEM vazamento de dados
    """
    
    def __init__(self, filepath, skiprows=8):
        super().__init__(filepath, skiprows)
        
    def remove_data_leakage(self, df):
        """
        Remove todas as variáveis que causam vazamento de dados
        """
        print("🚨 REMOVENDO VAZAMENTO DE DADOS...")
        
        # Variáveis que são vazamento direto
        leakage_vars = [
            'tx_promocao_EM',       # Taxa de promoção (diretamente relacionada)
            'tx_repetencia_EM',     # Taxa de repetência (complementar à evasão)  
            'tx_evasao_1_ano_EM',   # Evasão por ano (componente do target)
            'tx_evasao_2_ano_EM',   # Evasão por ano (componente do target)
            'tx_evasao_3_ano_EM',   # Evasão por ano (componente do target)
            'tx_aprovacao_EM',      # Taxa de aprovação (inverso da evasão)
            'tx_abandono_EM',       # Taxa de abandono (similar à evasão)
            'tx_migracao_eja_EM'    # Migração EJA (pode ser resultado da evasão)
        ]
        
        # Remove variáveis de vazamento
        found_leakage = [var for var in leakage_vars if var in df.columns]
        
        if found_leakage:
            print(f"   Removendo {len(found_leakage)} variáveis de vazamento:")
            for var in found_leakage:
                print(f"   - {var}")
            df = df.drop(columns=found_leakage)
        else:
            print("   ✅ Nenhuma variável de vazamento encontrada")
            
        return df
    
    def get_clean_features(self):
        """
        Lista de features limpas (sem vazamento) baseadas na análise
        """
        return [
            # Features geográficas (controladas)
            'NO_REGIAO', 'NO_UF',
            
            # INSE (Índice Socioeconômico) - VÁLIDAS
            'MEDIA_INSE', 'PC_NIVEL_1', 'PC_NIVEL_2', 'PC_NIVEL_3', 
            'PC_NIVEL_4', 'PC_NIVEL_5', 'PC_NIVEL_6', 'PC_NIVEL_7',
            
            # IDEB e Qualidade (anos anteriores) - VÁLIDAS se de períodos anteriores
            'VL_OBSERVADO_2021',      # IDEB observado
            'VL_PROJECAO_2021',       # IDEB projetado
            'VL_NOTA_MATEMATICA_2021', # Nota matemática
            'VL_NOTA_PORTUGUES_2021',  # Nota português
            'VL_NOTA_MEDIA_2021',      # Nota média
            
            # Indicadores de Risco (TDI) - VÁLIDAS
            'RISCO_PEDAGOGICO_TDI_ATU',
            'RISCO_INFRA_TDI_NET', 
            'RISCO_SOCIAL_TDI_PIB',
            'RISCO_GOVERNANCA_IDH',
            
            # Indicadores Socioeconômicos (IDH) - VÁLIDAS
            'ADH_IDHM', 'ADH_IDHM_E', 'ADH_IDHM_L', 'ADH_IDHM_R',
            'ADH_INDICE_GINI', 'ADH_RENDA_PER_CAPITA',
            'ADH_EXPECTATIVA_ANOS_ESTUDO', 'ADH_TX_ATRASO_2_FUNDAMENTAL',
            'ADH_TX_ANALFABETISMO_25_MAIS', 'ADH_PROP_POBREZA_EXTREMA',
            'ADH_PROP_VULNER_POBREZA', 'ADH_PERC_POPULACAO_RURAL',
            
            # Demografia e Raça - VÁLIDAS
            'RACA_PERC_PRETA_PARDA', 'RACA_PERC_INDIGENA',
            'CENSO_PERC_HOMENS', 'CENSO_PERC_MULHERES',
            
            # Indicadores Educacionais Estruturais - VÁLIDAS
            'MED_CAT_0_dsu',  # Docentes com superior
            'MED_CAT_0_tdi',  # Indicador TDI geral
        ]
    
    def prepare_data_clean(self, df, test_size=0.2, random_state=42):
        """
        Preparação de dados SEM vazamento
        """
        print("=" * 60)
        print("PREPARAÇÃO DE DADOS - VERSÃO LIMPA (SEM VAZAMENTO)")
        print("=" * 60)
        
        # 1. Remove vazamento de dados PRIMEIRO
        df = self.remove_data_leakage(df)
        
        # 2. Define target
        target = 'tx_evasao_total_EM'
        if target not in df.columns:
            raise ValueError(f"Target '{target}' não encontrado")
        
        # Remove linhas com target ausente
        original_len = len(df)
        df = df.dropna(subset=[target])
        print(f"Removidas {original_len - len(df)} linhas por target ausente")
        y = df[target]
        
        # 3. Seleciona features limpas
        clean_features = self.get_clean_features()
        available_features = [f for f in clean_features if f in df.columns]
        
        print(f"\\nFeatures disponíveis: {len(available_features)} de {len(clean_features)} solicitadas")
        print("Features não encontradas:", [f for f in clean_features if f not in df.columns])
        
        X = df[available_features].copy()
        
        # 4. One-hot encoding para categóricas
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print(f"Aplicando One-Hot Encoding: {categorical_cols}")
            X = pd.get_dummies(X, columns=categorical_cols, dummy_na=True)
        
        # Salva nomes das features
        self.feature_names = X.columns.tolist()
        print(f"Total de features após processamento: {len(self.feature_names)}")
        
        # 5. Divisão treino/teste
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=pd.qcut(y, q=5, labels=False)
        )
        
        # 6. Tratamento de missing values
        from sklearn.impute import SimpleImputer
        print("Imputação com mediana...")
        imputer = SimpleImputer(strategy='median')
        
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        X_train = pd.DataFrame(X_train_imputed, columns=self.feature_names, index=X_train.index)
        X_test = pd.DataFrame(X_test_imputed, columns=self.feature_names, index=X_test.index)
        
        # 7. Normalização
        from sklearn.preprocessing import StandardScaler
        print("Normalização com StandardScaler...")
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\\n✅ PREPARAÇÃO CONCLUÍDA (SEM VAZAMENTO)")
        print(f"Shape treino: {X_train_scaled.shape}")
        print(f"Shape teste: {X_test_scaled.shape}")
        print(f"Target - min: {y.min():.2f}, max: {y.max():.2f}, média: {y.mean():.2f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

def main_clean():
    """
    Pipeline LIMPO sem vazamento de dados
    """
    print("🧹 PIPELINE LIMPO - SEM VAZAMENTO DE DADOS")
    print("=" * 60)
    
    # Carrega dados
    clean_loader = CleanDataLoader('data/TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx')
    
    if os.path.exists('data/data_combined.csv'):
        print("Carregando dados combinados...")
        df = pd.read_csv('data/data_combined.csv', encoding='utf-8-sig')
    else:
        print("❌ Arquivo combinado não encontrado. Execute main.py primeiro.")
        return
    
    print(f"Dados originais: {df.shape}")
    
    # Prepara dados SEM vazamento
    X_train, X_test, y_train, y_test = clean_loader.prepare_data_clean(df)
    
    results = {}
    
    # Testa Random Forest
    print("\\n" + "="*50)
    print("🌲 RANDOM FOREST (SEM VAZAMENTO)")
    print("="*50)
    
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    
    # Mostra feature importance
    if hasattr(clean_loader, 'feature_names'):
        importance_df = rf_model.get_feature_importances(clean_loader.feature_names)
        print("\\nTop 10 Features Mais Importantes:")
        print(importance_df.head(10).to_string(index=False))
    
    results['Random Forest'] = {
        'R²': rf_metrics['r2'],
        'MAE': rf_metrics['mae'], 
        'MSE': rf_metrics['mse']
    }
    
    # Testa MLP
    print("\\n" + "="*50)
    print("🧠 MLP (SEM VAZAMENTO)")
    print("="*50)
    
    mlp_model = MLPModel(input_dim=X_train.shape[1])
    mlp_history = mlp_model.train(X_train, y_train, epochs=100)
    mlp_predictions = mlp_model.predict(X_test)
    mlp_loss, mlp_mae = mlp_model.evaluate(X_test, y_test)
    
    # Calcula R² para MLP
    from sklearn.metrics import r2_score
    mlp_r2 = r2_score(y_test, mlp_predictions)
    
    results['MLP'] = {
        'R²': mlp_r2,
        'MAE': mlp_mae,
        'MSE': mlp_loss
    }
    
    # Resultados finais
    print("\\n" + "="*60)
    print("📊 RESULTADOS FINAIS (SEM VAZAMENTO)")
    print("="*60)
    
    for model, metrics in results.items():
        print(f"\\n{model}:")
        print(f"  R²:  {metrics['R²']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  MSE: {metrics['MSE']:.4f}")
    
    # Comparação com resultados anteriores (com vazamento)
    print("\\n⚠️  COMPARAÇÃO:")
    print("   Resultados anteriores (COM vazamento): R² ≈ 0.95-0.98")
    print(f"   Resultados atuais (SEM vazamento): R² ≈ {results['Random Forest']['R²']:.3f}-{results['MLP']['R²']:.3f}")
    print("   ✅ Agora os resultados são realistas e confiáveis!")
    
    # Salva resultados limpos
    results_df = pd.DataFrame(results).T
    results_df.to_csv('resultados_limpos_sem_vazamento.csv')
    print(f"\\n📁 Resultados salvos em: resultados_limpos_sem_vazamento.csv")
    
    return results

if __name__ == "__main__":
    results = main_clean()