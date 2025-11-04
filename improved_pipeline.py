import os
import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.models.mlp_model import MLPModel
from src.models.random_forest_model import RandomForestModel
from src.visualization.plots import ModelVisualizer

class ImprovedDataLoader(DataLoader):
    """
    DataLoader melhorado com feature selection e engineering
    """
    
    def __init__(self, filepath, skiprows=8):
        super().__init__(filepath, skiprows)
        self.selected_features = None
        
    def get_consensus_features(self):
        """
        Retorna as features com consenso dos diferentes métodos de seleção
        """
        return [
            'RISCO_PEDAGOGICO_TDI_ATU',
            'RISCO_INFRA_TDI_NET', 
            'VL_OBSERVADO_2021',
            'PC_NIVEL_5',
            'PC_NIVEL_7',
            'MEDIA_INSE',
            'PC_NIVEL_6',
            'PC_NIVEL_3',
            'PC_NIVEL_2',
            'PC_NIVEL_4',
            'PC_NIVEL_1',
            'NO_UF_PA',
            'RISCO_GOVERNANCA_IDH',
            'MED_CAT_0_dsu',
            'ADH_EXPECTATIVA_ANOS_ESTUDO',
            'ADH_IDHM',
            'ADH_TX_ANALFABETISMO_25_MAIS',
            'ADH_TX_ATRASO_2_FUNDAMENTAL',
            'NO_REGIAO_Sudeste',
            'VL_PROJECAO_2021'
        ]
    
    def remove_highly_correlated_features(self, df, threshold=0.95):
        """
        Remove features altamente correlacionadas
        """
        print(f"Removendo features com correlação > {threshold}...")
        
        # Calcula correlação apenas para features numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Encontra pares altamente correlacionados
        upper_tri = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        # Identifica features para remover
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        print(f"Features removidas por alta correlação: {len(to_drop)}")
        if to_drop:
            print("Primeiras 10:", to_drop[:10])
        
        return df.drop(columns=to_drop, errors='ignore')
    
    def create_feature_interactions(self, df):
        """
        Cria features de interação relevantes
        """
        print("Criando features de interação...")
        
        # Interações importantes baseadas na análise
        interactions = []
        
        # Ratio INSE vs Infraestrutura
        if 'MEDIA_INSE' in df.columns and 'RISCO_INFRA_TDI_NET' in df.columns:
            df['INSE_INFRA_RATIO'] = df['MEDIA_INSE'] / (df['RISCO_INFRA_TDI_NET'] + 1e-8)
            interactions.append('INSE_INFRA_RATIO')
        
        # Indicador composto de risco
        risk_cols = [col for col in df.columns if 'RISCO_' in col]
        if len(risk_cols) >= 2:
            df['RISCO_COMPOSTO'] = df[risk_cols].mean(axis=1)
            interactions.append('RISCO_COMPOSTO')
        
        # Ratio IDH vs Analfabetismo
        if 'ADH_IDHM' in df.columns and 'ADH_TX_ANALFABETISMO_25_MAIS' in df.columns:
            df['IDH_ANALFABETISMO_RATIO'] = df['ADH_IDHM'] / (df['ADH_TX_ANALFABETISMO_25_MAIS'] + 1e-8)
            interactions.append('IDH_ANALFABETISMO_RATIO')
        
        # Indicador socioeconômico composto
        socio_cols = [col for col in df.columns if 'PC_NIVEL_' in col]
        if len(socio_cols) >= 3:
            # Peso maior para níveis mais altos
            weights = {f'PC_NIVEL_{i}': i for i in range(1, 8)}
            weighted_sum = sum(df[col] * weights.get(col, 1) for col in socio_cols if col in df.columns)
            df['INSE_WEIGHTED'] = weighted_sum / len(socio_cols)
            interactions.append('INSE_WEIGHTED')
        
        print(f"Features de interação criadas: {interactions}")
        return df, interactions
    
    def remove_outliers_iqr(self, df, target_col, factor=2.0):
        """
        Remove outliers usando método IQR mais conservador
        """
        print(f"Removendo outliers do target com fator IQR = {factor}...")
        
        if target_col not in df.columns:
            return df
        
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        before = len(df)
        df_clean = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
        after = len(df_clean)
        
        print(f"Outliers removidos: {before - after} ({(before-after)/before*100:.1f}%)")
        
        return df_clean
    
    def prepare_data_improved(self, df, use_feature_selection=True, 
                            remove_correlated=True, create_interactions=True, 
                            remove_outliers=True, test_size=0.2, random_state=42):
        """
        Versão melhorada do prepare_data com todas as otimizações
        """
        print("=== PREPARAÇÃO DE DADOS MELHORADA ===")
        
        # 1. Remove outliers primeiro
        if remove_outliers:
            target = 'tx_evasao_total_EM'
            if target in df.columns:
                df = self.remove_outliers_iqr(df, target, factor=2.0)
        
        # 2. Remove features altamente correlacionadas
        if remove_correlated:
            df = self.remove_highly_correlated_features(df, threshold=0.95)
        
        # 3. Cria features de interação
        interaction_features = []
        if create_interactions:
            df, interaction_features = self.create_feature_interactions(df)
        
        # 4. Define target e limpa NaNs
        target = 'tx_evasao_total_EM'
        if target not in df.columns:
            raise ValueError(f"Coluna alvo '{target}' não encontrada")
        
        original_len = len(df)
        df = df.dropna(subset=[target])
        print(f"Removidas {original_len - len(df)} linhas por alvo ausente.")
        y = df[target]
        
        # 5. Feature selection
        if use_feature_selection:
            print("Usando feature selection baseada em consenso...")
            consensus_features = self.get_consensus_features()
            
            # Adiciona features de interação criadas
            consensus_features.extend(interaction_features)
            
            # Filtra para features que existem no df
            available_features = [f for f in consensus_features if f in df.columns]
            
            # Adiciona algumas features categóricas importantes se ainda não incluídas
            categorical_important = ['NO_REGIAO', 'NO_UF']
            for cat in categorical_important:
                if cat in df.columns and cat not in available_features:
                    available_features.append(cat)
            
            print(f"Features selecionadas: {len(available_features)}")
            X = df[available_features].copy()
        else:
            # Usa abordagem original
            feature_list = [col for col in df.columns if col != target and not col.startswith('tx_evasao')]
            available_features = [f for f in feature_list if f in df.columns]
            X = df[available_features].copy()
        
        # 6. One-hot encoding
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print(f"Aplicando One-Hot Encoding em: {categorical_cols}")
            X = pd.get_dummies(X, columns=categorical_cols, dummy_na=True)
        
        # Salva nomes das features
        self.feature_names = X.columns.tolist()
        print(f"Total de features após processamento: {len(self.feature_names)}")
        
        # 7. Divisão treino/teste
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 8. Imputação
        from sklearn.impute import SimpleImputer
        print("Preenchendo valores ausentes com mediana...")
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train)
        
        X_train_imputed = imputer.transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        X_train = pd.DataFrame(X_train_imputed, columns=self.feature_names, index=X_train.index)
        X_test = pd.DataFrame(X_test_imputed, columns=self.feature_names, index=X_test.index)
        
        # 9. Normalização
        print("Normalizando dados...")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Preparação de dados concluída!")
        print(f"Shape treino: {X_train_scaled.shape}")
        print(f"Shape teste: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

def main_improved():
    """
    Pipeline principal melhorado
    """
    print("=== PIPELINE MELHORADO DE ANÁLISE ===\n")
    
    # Carrega dados
    improved_loader = ImprovedDataLoader('data/TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx')
    
    if os.path.exists('data/data_combined.csv'):
        print("Carregando dados combinados...")
        df = pd.read_csv('data/data_combined.csv', encoding='utf-8-sig')
    else:
        print("Arquivo combinado não encontrado. Execute main.py primeiro.")
        return
    
    print(f"Dados originais: {df.shape}")
    
    # Experimenta diferentes configurações
    configurations = [
        {
            'name': 'Baseline (Original)',
            'use_feature_selection': False,
            'remove_correlated': False,
            'create_interactions': False,
            'remove_outliers': False
        },
        {
            'name': 'Feature Selection Apenas',
            'use_feature_selection': True,
            'remove_correlated': False,
            'create_interactions': False,
            'remove_outliers': False
        },
        {
            'name': 'Completa Otimizada',
            'use_feature_selection': True,
            'remove_correlated': True,
            'create_interactions': True,
            'remove_outliers': True
        },
        {
            'name': 'Sem Outliers',
            'use_feature_selection': True,
            'remove_correlated': True,
            'create_interactions': True,
            'remove_outliers': False
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"TESTANDO CONFIGURAÇÃO: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Prepara dados com a configuração atual
            X_train, X_test, y_train, y_test = improved_loader.prepare_data_improved(
                df.copy(),
                use_feature_selection=config['use_feature_selection'],
                remove_correlated=config['remove_correlated'],
                create_interactions=config['create_interactions'],
                remove_outliers=config['remove_outliers']
            )
            
            # Testa Random Forest
            print(f"\n--- Random Forest ({config['name']}) ---")
            rf_model = RandomForestModel()
            rf_model.train(X_train, y_train)
            rf_metrics = rf_model.evaluate(X_test, y_test)
            
            # Testa MLP
            print(f"\n--- MLP ({config['name']}) ---")
            mlp_model = MLPModel(input_dim=X_train.shape[1])
            mlp_history = mlp_model.train(X_train, y_train, epochs=50)
            mlp_predictions = mlp_model.predict(X_test)
            mlp_loss, mlp_mae = mlp_model.evaluate(X_test, y_test)
            
            # Calcula R² para MLP
            from sklearn.metrics import r2_score
            mlp_r2 = r2_score(y_test, mlp_predictions)
            
            # Armazena resultados
            result = {
                'configuration': config['name'],
                'features_count': X_train.shape[1],
                'samples_count': len(X_train),
                'rf_r2': rf_metrics['r2'],
                'rf_mae': rf_metrics['mae'],
                'rf_mse': rf_metrics['mse'],
                'mlp_r2': mlp_r2,
                'mlp_mae': mlp_mae,
                'mlp_mse': mlp_loss
            }
            
            results.append(result)
            
            print(f"\n📊 RESULTADOS ({config['name']}):")
            print(f"   Features: {X_train.shape[1]}")
            print(f"   Amostras: {len(X_train)}")
            print(f"   Random Forest R²: {rf_metrics['r2']:.4f}")
            print(f"   MLP R²: {mlp_r2:.4f}")
            print(f"   RF MAE: {rf_metrics['mae']:.4f}, MLP MAE: {mlp_mae:.4f}")
            
        except Exception as e:
            print(f"❌ Erro na configuração {config['name']}: {e}")
            continue
    
    # Compara resultados
    print(f"\n{'='*80}")
    print("COMPARAÇÃO FINAL DOS RESULTADOS")
    print(f"{'='*80}")
    
    if not results:
        print("❌ Nenhum resultado válido obtido.")
        return None
    
    results_df = pd.DataFrame(results)
    
    print("\nR² Score Comparison:")
    if 'rf_r2' in results_df.columns:
        print(results_df[['configuration', 'features_count', 'rf_r2']].to_string(index=False))
        if 'mlp_r2' in results_df.columns:
            print("\nComparação incluindo MLP:")
            print(results_df[['configuration', 'features_count', 'rf_r2', 'mlp_r2']].to_string(index=False))
    
    print("\nMAE Comparison:")
    if 'rf_mae' in results_df.columns:
        mae_cols = ['configuration', 'rf_mae']
        if 'mlp_mae' in results_df.columns:
            mae_cols.append('mlp_mae')
        print(results_df[mae_cols].to_string(index=False))
    
    # Identifica melhor configuração
    if 'rf_r2' in results_df.columns and len(results_df) > 0:
        best_rf = results_df.loc[results_df['rf_r2'].idxmax()]
        print(f"\n🏆 MELHOR RESULTADO Random Forest:")
        print(f"   Configuração: {best_rf['configuration']}")
        print(f"   R²: {best_rf['rf_r2']:.4f}")
        print(f"   MAE: {best_rf['rf_mae']:.4f}")
        print(f"   Features: {best_rf['features_count']}")
        
        if 'mlp_r2' in results_df.columns:
            best_mlp = results_df.loc[results_df['mlp_r2'].idxmax()]
            print(f"\n🏆 MELHOR RESULTADO MLP:")
            print(f"   Configuração: {best_mlp['configuration']}")
            print(f"   R²: {best_mlp['mlp_r2']:.4f}")
            print(f"   MAE: {best_mlp['mlp_mae']:.4f}")
            print(f"   Features: {best_mlp['features_count']}")
    
    # Salva resultados
    results_df.to_csv('improvement_results.csv', index=False)
    print(f"\n📁 Resultados salvos em: improvement_results.csv")
    
    return results_df

if __name__ == "__main__":
    results = main_improved()