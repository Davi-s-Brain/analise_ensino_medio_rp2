import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from src.data.loader import DataLoader
from src.models.mlp_model import MLPModel
from src.models.random_forest_model import RandomForestModel

class OptimizedDataLoader(DataLoader):
    """
    DataLoader otimizado - Remove vazamento E variáveis de baixa importância
    """
    
    def __init__(self, filepath, skiprows=8):
        super().__init__(filepath, skiprows)
        
    def remove_data_leakage(self, df):
        """
        Remove todas as variáveis que causam vazamento de dados
        """
        print("🚨 REMOVENDO VAZAMENTO DE DADOS...")
        
        leakage_vars = [
            'tx_promocao_EM', 'tx_repetencia_EM', 'tx_evasao_1_ano_EM', 
            'tx_evasao_2_ano_EM', 'tx_evasao_3_ano_EM', 'tx_aprovacao_EM', 
            'tx_abandono_EM', 'tx_migracao_eja_EM'
        ]
        
        found_leakage = [var for var in leakage_vars if var in df.columns]
        
        if found_leakage:
            print(f"   Removendo {len(found_leakage)} variáveis de vazamento")
            df = df.drop(columns=found_leakage)
        
        return df
    
    def remove_low_importance_features(self, df):
        """
        Remove variáveis com baixa importância (principalmente geográficas)
        """
        print("🔧 REMOVENDO VARIÁVEIS DE BAIXA IMPORTÂNCIA...")
        
        # Variáveis geográficas específicas (baixa importância)
        low_importance_geo = [
            'NO_UF_MT', 'NO_UF_RN', 'NO_UF_BA', 'NO_UF_SC', 'NO_UF_PI', 
            'NO_UF_GO', 'NO_UF_RS', 'NO_UF_PR', 'NO_UF_MG', 'NO_UF_ES',
            'NO_UF_SP', 'NO_UF_RJ', 'NO_UF_CE', 'NO_UF_PE', 'NO_UF_AL', 
            'NO_UF_SE', 'NO_UF_AM', 'NO_UF_MS', 'NO_UF_AC', 'NO_UF_AP',
            'NO_UF_RO', 'NO_UF_RR', 'NO_UF_TO', 'NO_UF_DF', 'NO_UF_MA'
        ]
        
        # Remove variáveis encontradas
        found_low_importance = [var for var in low_importance_geo if var in df.columns]
        
        if found_low_importance:
            print(f"   Removendo {len(found_low_importance)} variáveis geográficas de baixa importância")
            df = df.drop(columns=found_low_importance)
        
        return df
    
    def get_high_importance_features(self):
        """
        Lista apenas features com alta importância (> 1.5%)
        """
        return [
            # Features geográficas importantes (mantidas)
            'NO_REGIAO',  # Região é mais importante que estados individuais
            
            # INSE (Índice Socioeconômico) - ALTA IMPORTÂNCIA
            'MEDIA_INSE', 'PC_NIVEL_1', 'PC_NIVEL_2', 'PC_NIVEL_3', 
            'PC_NIVEL_4', 'PC_NIVEL_5', 'PC_NIVEL_6', 'PC_NIVEL_7',
            
            # IDEB e Qualidade - ALTA IMPORTÂNCIA
            'VL_OBSERVADO_2021',      # IDEB observado (3.8%)
            'VL_PROJECAO_2021',       # IDEB projetado
            'VL_NOTA_MATEMATICA_2021', 
            'VL_NOTA_PORTUGUES_2021',  
            'VL_NOTA_MEDIA_2021',      
            
            # Indicadores de Risco (TDI) - MUITO ALTA IMPORTÂNCIA
            'RISCO_PEDAGOGICO_TDI_ATU',  # 4.9%
            'RISCO_INFRA_TDI_NET',       # 3.1%
            'RISCO_SOCIAL_TDI_PIB',      # 2.5%
            'RISCO_GOVERNANCA_IDH',      # 1.8%
            
            # Indicadores Socioeconômicos (IDH) - ALTA IMPORTÂNCIA
            'ADH_IDHM', 'ADH_IDHM_E', 'ADH_IDHM_L', 'ADH_IDHM_R',
            'ADH_EXPECTATIVA_ANOS_ESTUDO',    # 2.3%
            'ADH_TX_ATRASO_2_FUNDAMENTAL',    # 1.7%
            'ADH_TX_ANALFABETISMO_25_MAIS',   # 1.6%
            'ADH_PROP_POBREZA_EXTREMA',
            'ADH_PROP_VULNER_POBREZA', 
            'ADH_PERC_POPULACAO_RURAL',
            'ADH_RENDA_PER_CAPITA',
            'ADH_INDICE_GINI',
            
            # Demografia e Raça - ALTA IMPORTÂNCIA
            'RACA_PERC_PRETA_PARDA',    # 2.7%
            'RACA_PERC_INDIGENA',
            
            # Indicadores Educacionais Estruturais - MUITO ALTA IMPORTÂNCIA
            'MED_CAT_0_dsu',  # Docentes com superior (1.7%)
            'MED_CAT_0_tdi',  # Indicador TDI geral (25.6% - MAIS IMPORTANTE!)
        ]
    
    def prepare_data_optimized(self, df, test_size=0.2, random_state=42):
        """
        Preparação de dados OTIMIZADA - Remove vazamento E baixa importância
        """
        print("=" * 70)
        print("PREPARAÇÃO DE DADOS OTIMIZADA - APENAS FEATURES IMPORTANTES")
        print("=" * 70)
        
        # 1. Remove vazamento de dados
        df = self.remove_data_leakage(df)
        
        # 2. Remove variáveis de baixa importância
        df = self.remove_low_importance_features(df)
        
        # 3. Define target
        target = 'tx_evasao_total_EM'
        if target not in df.columns:
            raise ValueError(f"Target '{target}' não encontrado")
        
        original_len = len(df)
        df = df.dropna(subset=[target])
        print(f"Removidas {original_len - len(df)} linhas por target ausente")
        y = df[target]
        
        # 4. Seleciona apenas features de alta importância
        high_importance_features = self.get_high_importance_features()
        available_features = [f for f in high_importance_features if f in df.columns]
        
        print(f"\\nFeatures de alta importância: {len(available_features)} de {len(high_importance_features)} solicitadas")
        missing_features = [f for f in high_importance_features if f not in df.columns]
        if missing_features:
            print(f"Features não encontradas: {missing_features}")
        
        X = df[available_features].copy()
        
        # 5. One-hot encoding para categóricas
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print(f"Aplicando One-Hot Encoding: {categorical_cols}")
            X = pd.get_dummies(X, columns=categorical_cols, dummy_na=True)
        
        # Salva nomes das features
        self.feature_names = X.columns.tolist()
        print(f"Total de features otimizadas: {len(self.feature_names)}")
        
        # 6. Divisão treino/teste estratificada
        from sklearn.model_selection import train_test_split
        # Estratifica por quintis do target
        y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y_binned
        )
        
        # 7. Tratamento de missing values
        from sklearn.impute import SimpleImputer
        print("Imputação com mediana...")
        imputer = SimpleImputer(strategy='median')
        
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        X_train = pd.DataFrame(X_train_imputed, columns=self.feature_names, index=X_train.index)
        X_test = pd.DataFrame(X_test_imputed, columns=self.feature_names, index=X_test.index)
        
        # 8. Normalização robusta
        from sklearn.preprocessing import RobustScaler
        print("Normalização com RobustScaler...")
        scaler = RobustScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\\n✅ PREPARAÇÃO OTIMIZADA CONCLUÍDA")
        print(f"Shape treino: {X_train_scaled.shape}")
        print(f"Shape teste: {X_test_scaled.shape}")
        print(f"Target - min: {y.min():.2f}, max: {y.max():.2f}, média: {y.mean():.2f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df[available_features + [target]]

def create_plotly_visualizations(loader, X_train, X_test, y_train, y_test, df_features):
    """
    Cria visualizações com Plotly
    """
    print("\\n📊 CRIANDO VISUALIZAÇÕES COM PLOTLY...")
    
    # 1. Treina modelo para obter feature importance
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    importance_df = rf_model.get_feature_importances(loader.feature_names)
    
    # Metrics
    rf_metrics = rf_model.evaluate(X_test, y_test)
    
    # 2. GRÁFICO 1: Feature Importance
    fig_importance = go.Figure()
    
    top_15 = importance_df.head(15)
    
    fig_importance.add_trace(go.Bar(
        y=top_15['feature'],
        x=top_15['importance'],
        orientation='h',
        marker=dict(
            color=top_15['importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importância")
        ),
        text=[f'{val:.3f}' for val in top_15['importance']],
        textposition='auto',
        textfont=dict(color='white', size=10)
    ))
    
    fig_importance.update_layout(
        title=dict(
            text='Top 15 Features Mais Importantes (Random Forest)',
            x=0.5,
            font=dict(size=16, color='darkblue')
        ),
        xaxis_title='Importância',
        yaxis_title='Features',
        height=600,
        template='plotly_white',
        font=dict(size=12)
    )
    
    fig_importance.write_html('feature_importance_plotly.html')
    print("✅ Gráfico de importância salvo: feature_importance_plotly.html")
    
    # 3. GRÁFICO 2: Distribuição do Target
    target_col = 'tx_evasao_total_EM'
    target_data = df_features[target_col]
    
    fig_target = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Histograma', 'Box Plot', 'Densidade', 'Estatísticas'),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Histograma
    fig_target.add_trace(
        go.Histogram(
            x=target_data,
            nbinsx=50,
            name='Distribuição',
            marker_color='skyblue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Box Plot
    fig_target.add_trace(
        go.Box(
            y=target_data,
            name='Box Plot',
            marker_color='lightcoral',
            boxmean=True
        ),
        row=1, col=2
    )
    
    # Densidade
    fig_target.add_trace(
        go.Histogram(
            x=target_data,
            histnorm='probability density',
            nbinsx=30,
            name='Densidade',
            marker_color='lightgreen',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Estatísticas (como texto)
    stats_text = f'''
    Média: {target_data.mean():.2f}%<br>
    Mediana: {target_data.median():.2f}%<br>
    Desvio Padrão: {target_data.std():.2f}%<br>
    Mínimo: {target_data.min():.2f}%<br>
    Máximo: {target_data.max():.2f}%<br>
    <br>
    Q1: {target_data.quantile(0.25):.2f}%<br>
    Q3: {target_data.quantile(0.75):.2f}%<br>
    IQR: {target_data.quantile(0.75) - target_data.quantile(0.25):.2f}%<br>
    <br>
    Outliers (> Q3 + 1.5*IQR): {len(target_data[target_data > target_data.quantile(0.75) + 1.5*(target_data.quantile(0.75) - target_data.quantile(0.25))])}<br>
    Assimetria: {target_data.skew():.3f}
    '''
    
    fig_target.add_annotation(
        text=stats_text,
        xref="x domain", yref="y domain",
        x=0.1, y=0.9,
        showarrow=False,
        font=dict(size=11),
        bgcolor="lightyellow",
        bordercolor="gray",
        borderwidth=1,
        row=2, col=2
    )
    
    fig_target.update_layout(
        title=dict(
            text='Análise da Taxa de Evasão do Ensino Médio',
            x=0.5,
            font=dict(size=16, color='darkblue')
        ),
        height=700,
        showlegend=False,
        template='plotly_white'
    )
    
    fig_target.write_html('analise_target_plotly.html')
    print("✅ Análise do target salva: analise_target_plotly.html")
    
    # 4. GRÁFICO 3: Performance dos Modelos
    fig_performance = go.Figure()
    
    models = ['Random Forest']
    r2_values = [rf_metrics['r2']]
    mae_values = [rf_metrics['mae']]
    
    fig_performance.add_trace(go.Bar(
        x=models,
        y=r2_values,
        name='R² Score',
        marker_color='forestgreen',
        text=[f'{val:.3f}' for val in r2_values],
        textposition='auto',
        yaxis='y'
    ))
    
    fig_performance.add_trace(go.Bar(
        x=models,
        y=mae_values,
        name='MAE',
        marker_color='coral',
        text=[f'{val:.2f}' for val in mae_values],
        textposition='auto',
        yaxis='y2'
    ))
    
    fig_performance.update_layout(
        title=dict(
            text='Performance do Modelo (Dados Otimizados)',
            x=0.5,
            font=dict(size=16, color='darkblue')
        ),
        yaxis=dict(title='R² Score', side='left'),
        yaxis2=dict(title='MAE', side='right', overlaying='y'),
        template='plotly_white',
        height=500
    )
    
    fig_performance.write_html('performance_modelo_plotly.html')
    print("✅ Performance do modelo salva: performance_modelo_plotly.html")
    
    # 5. GRÁFICO 4: Correlações com Target (top features)
    numeric_features = df_features.select_dtypes(include=[np.number]).columns
    correlations = []
    
    for col in numeric_features:
        if col != target_col:
            corr = df_features[col].corr(df_features[target_col])
            if not np.isnan(corr):
                correlations.append({'feature': col, 'correlation': corr, 'abs_correlation': abs(corr)})
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False).head(10)
    
    fig_corr = go.Figure()
    
    colors = ['green' if x > 0 else 'red' for x in corr_df['correlation']]
    
    fig_corr.add_trace(go.Bar(
        y=corr_df['feature'],
        x=corr_df['correlation'],
        orientation='h',
        marker=dict(color=colors, opacity=0.7),
        text=[f'{val:.3f}' for val in corr_df['correlation']],
        textposition='auto'
    ))
    
    fig_corr.update_layout(
        title=dict(
            text='Top 10 Correlações com Taxa de Evasão',
            x=0.5,
            font=dict(size=16, color='darkblue')
        ),
        xaxis_title='Correlação',
        yaxis_title='Features',
        height=600,
        template='plotly_white'
    )
    
    fig_corr.write_html('correlacoes_target_plotly.html')
    print("✅ Correlações salvas: correlacoes_target_plotly.html")
    
    return {
        'r2': rf_metrics['r2'],
        'mae': rf_metrics['mae'],
        'mse': rf_metrics['mse'],
        'n_features': len(loader.feature_names),
        'importance_df': importance_df
    }

def main_optimized():
    """
    Pipeline principal otimizado com Plotly
    """
    print("🎯 PIPELINE OTIMIZADO COM PLOTLY - APENAS FEATURES IMPORTANTES")
    print("=" * 70)
    
    # Carrega dados
    optimized_loader = OptimizedDataLoader('data/TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx')
    
    if os.path.exists('data/data_combined.csv'):
        print("Carregando dados combinados...")
        df = pd.read_csv('data/data_combined.csv', encoding='utf-8-sig')
    else:
        print("❌ Arquivo combinado não encontrado. Execute main.py primeiro.")
        return
    
    print(f"Dados originais: {df.shape}")
    
    # Prepara dados otimizados
    X_train, X_test, y_train, y_test, df_features = optimized_loader.prepare_data_optimized(df)
    
    # Cria visualizações
    results = create_plotly_visualizations(optimized_loader, X_train, X_test, y_train, y_test, df_features)
    
    # Resultados finais
    print("\\n" + "="*70)
    print("📊 RESULTADOS FINAIS (PIPELINE OTIMIZADO)")
    print("="*70)
    print(f"R² Score: {results['r2']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"Features utilizadas: {results['n_features']}")
    
    print("\\n📁 ARQUIVOS PLOTLY CRIADOS:")
    print("   - feature_importance_plotly.html")
    print("   - analise_target_plotly.html")
    print("   - performance_modelo_plotly.html")
    print("   - correlacoes_target_plotly.html")
    
    # Salva resultados
    results_summary = pd.DataFrame([results])
    results_summary.to_csv('resultados_otimizados_plotly.csv', index=False)
    
    # Salva feature importance
    results['importance_df'].to_csv('feature_importance_otimizada.csv', index=False)
    
    print("\\n💾 Dados salvos:")
    print("   - resultados_otimizados_plotly.csv")
    print("   - feature_importance_otimizada.csv")
    
    return results

if __name__ == "__main__":
    results = main_optimized()