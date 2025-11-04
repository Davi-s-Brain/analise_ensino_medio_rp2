import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import DataLoader
from src.models.random_forest_model import RandomForestModel
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor

def analyze_feature_importance():
    """
    Análise detalhada da importância das features para identificar as mais relevantes
    """
    
    # Carrega e prepara os dados
    print("Carregando e preparando dados...")
    data_loader = DataLoader('data/TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx')
    
    if os.path.exists('data/data_combined.csv'):
        df = pd.read_csv('data/data_combined.csv', encoding='utf-8-sig')
    else:
        print("Arquivo combinado não encontrado. Execute main.py primeiro.")
        return
    
    # Prepara os dados
    X_train_scaled, X_test_scaled, y_train, y_test = data_loader.prepare_data(df)
    feature_names = data_loader.feature_names
    
    print(f"Número de features após preparação: {len(feature_names)}")
    print(f"Shape do treino: {X_train_scaled.shape}")
    
    # 1. FEATURE IMPORTANCE DO RANDOM FOREST
    print("\n" + "="*60)
    print("1. ANÁLISE DE IMPORTÂNCIA - RANDOM FOREST")
    print("="*60)
    
    rf_model = RandomForestModel()
    rf_model.train(X_train_scaled, y_train)
    
    # Obtém a importância das features
    importance_df = rf_model.get_feature_importances(feature_names)
    
    print("Top 20 Features mais importantes (Random Forest):")
    print(importance_df.head(20))
    
    # Visualização
    plt.figure(figsize=(12, 10))
    top_20 = importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['importance'])
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Importância')
    plt.title('Top 20 Features - Random Forest Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ANÁLISE UNIVARIADA (F-SCORE)
    print("\n" + "="*60)
    print("2. ANÁLISE UNIVARIADA - F-SCORE")
    print("="*60)
    
    # SelectKBest com f_regression
    selector_f = SelectKBest(score_func=f_regression, k='all')
    selector_f.fit(X_train_scaled, y_train)
    
    f_scores = pd.DataFrame({
        'feature': feature_names,
        'f_score': selector_f.scores_
    }).sort_values('f_score', ascending=False)
    
    print("Top 20 Features com maior F-Score:")
    print(f_scores.head(20))
    
    # 3. MUTUAL INFORMATION
    print("\n" + "="*60)
    print("3. ANÁLISE - MUTUAL INFORMATION")
    print("="*60)
    
    # Mutual Information
    mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
    mi_df = pd.DataFrame({
        'feature': feature_names,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("Top 20 Features com maior Mutual Information:")
    print(mi_df.head(20))
    
    # 4. RECURSIVE FEATURE ELIMINATION (RFE)
    print("\n" + "="*60)
    print("4. RECURSIVE FEATURE ELIMINATION")
    print("="*60)
    
    # RFE com RandomForestRegressor
    rf_estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf_estimator, n_features_to_select=20, step=10)
    rfe.fit(X_train_scaled, y_train)
    
    rfe_features = pd.DataFrame({
        'feature': feature_names,
        'selected': rfe.support_,
        'ranking': rfe.ranking_
    }).sort_values('ranking')
    
    selected_features = rfe_features[rfe_features['selected']]['feature'].tolist()
    print(f"Top 20 Features selecionadas pelo RFE:")
    for i, feat in enumerate(selected_features, 1):
        print(f"{i:2d}. {feat}")
    
    # 5. ANÁLISE DE FEATURES COM BAIXA IMPORTÂNCIA
    print("\n" + "="*60)
    print("5. FEATURES COM BAIXA IMPORTÂNCIA")
    print("="*60)
    
    # Features com importância muito baixa no RF
    low_importance = importance_df[importance_df['importance'] < 0.001]
    print(f"Features com importância < 0.001 no Random Forest: {len(low_importance)}")
    if len(low_importance) > 0:
        print("Features a considerar para remoção:")
        print(low_importance[['feature', 'importance']].head(10))
    
    # 6. COMPARAÇÃO DOS MÉTODOS
    print("\n" + "="*60)
    print("6. COMPARAÇÃO DOS MÉTODOS")
    print("="*60)
    
    # Normaliza os scores para comparação
    importance_df['rf_norm'] = importance_df['importance'] / importance_df['importance'].max()
    f_scores['f_norm'] = f_scores['f_score'] / f_scores['f_score'].max()
    mi_df['mi_norm'] = mi_df['mutual_info'] / mi_df['mutual_info'].max()
    
    # Merge dos resultados
    comparison = importance_df[['feature', 'rf_norm']].copy()
    comparison = comparison.merge(f_scores[['feature', 'f_norm']], on='feature')
    comparison = comparison.merge(mi_df[['feature', 'mi_norm']], on='feature')
    
    # Score médio
    comparison['mean_score'] = (comparison['rf_norm'] + comparison['f_norm'] + comparison['mi_norm']) / 3
    comparison = comparison.sort_values('mean_score', ascending=False)
    
    print("Top 20 Features - Score Médio dos 3 métodos:")
    print(comparison[['feature', 'mean_score', 'rf_norm', 'f_norm', 'mi_norm']].head(20))
    
    # 7. RECOMENDAÇÕES DE FEATURE SELECTION
    print("\n" + "="*60)
    print("7. RECOMENDAÇÕES DE FEATURE SELECTION")
    print("="*60)
    
    # Features que aparecem no top 20 de pelo menos 2 métodos
    top_rf = set(importance_df.head(20)['feature'])
    top_f = set(f_scores.head(20)['feature'])
    top_mi = set(mi_df.head(20)['feature'])
    top_rfe = set(selected_features)
    
    # Intersection analysis
    rf_f_intersection = top_rf.intersection(top_f)
    rf_mi_intersection = top_rf.intersection(top_mi)
    f_mi_intersection = top_f.intersection(top_mi)
    all_three = top_rf.intersection(top_f).intersection(top_mi)
    
    print(f"Features no top 20 de Random Forest E F-Score: {len(rf_f_intersection)}")
    print(f"Features no top 20 de Random Forest E Mutual Info: {len(rf_mi_intersection)}")
    print(f"Features no top 20 de F-Score E Mutual Info: {len(f_mi_intersection)}")
    print(f"Features no top 20 de TODOS os 3 métodos: {len(all_three)}")
    
    # Features consensuais (aparecem em pelo menos 2 métodos)
    consensus_features = rf_f_intersection.union(rf_mi_intersection).union(f_mi_intersection)
    print(f"\nFeatures com consenso (top 20 em pelo menos 2 métodos): {len(consensus_features)}")
    
    consensus_list = list(consensus_features)
    consensus_list.sort()
    for i, feat in enumerate(consensus_list, 1):
        print(f"{i:2d}. {feat}")
    
    # 8. ANÁLISE DE GRUPOS DE FEATURES
    print("\n" + "="*60)
    print("8. ANÁLISE POR GRUPOS DE FEATURES")
    print("="*60)
    
    # Categoriza features por grupos
    feature_groups = {
        'Transição/Evasão': [f for f in feature_names if any(x in f.lower() for x in ['tx_', 'evasao', 'promocao', 'repetencia', 'aprovacao', 'abandono'])],
        'INSE/Socioeconômico': [f for f in feature_names if any(x in f.lower() for x in ['inse', 'pc_nivel', 'media_inse'])],
        'IDEB/Qualidade': [f for f in feature_names if any(x in f.lower() for x in ['ideb', 'vl_nota', 'vl_observado', 'vl_aprovacao'])],
        'Infraestrutura': [f for f in feature_names if any(x in f.lower() for x in ['in_', 'qt_salas', 'qt_doc', 'qt_funcionarios'])],
        'PIB/Econômico': [f for f in feature_names if any(x in f.lower() for x in ['pib', 'produto_interno', 'valor_adicionado', 'impostos'])],
        'IDH/Social': [f for f in feature_names if any(x in f.lower() for x in ['adh_', 'idhm', 'gini', 'pobreza', 'renda'])],
        'Demográfico': [f for f in feature_names if any(x in f.lower() for x in ['raca_', 'censo_', 'populacao'])],
        'Risco/Indicadores': [f for f in feature_names if any(x in f.lower() for x in ['risco_', 'med_cat', 'tdi', 'afd', 'ied'])],
        'Geográfico': [f for f in feature_names if any(x in f for x in ['NO_UF_', 'NO_REGIAO_'])]
    }
    
    # Calcula importância média por grupo
    group_importance = {}
    for group, features in feature_groups.items():
        group_features = [f for f in features if f in importance_df['feature'].values]
        if group_features:
            avg_importance = importance_df[importance_df['feature'].isin(group_features)]['importance'].mean()
            group_importance[group] = {
                'avg_importance': avg_importance,
                'count': len(group_features),
                'total_importance': importance_df[importance_df['feature'].isin(group_features)]['importance'].sum()
            }
    
    print("Importância média por grupo de features:")
    for group, stats in sorted(group_importance.items(), key=lambda x: x[1]['avg_importance'], reverse=True):
        print(f"{group:20s}: Média={stats['avg_importance']:.4f}, Count={stats['count']:3d}, Total={stats['total_importance']:.4f}")
    
    # Salva os resultados
    comparison.to_csv('feature_analysis_results.csv', index=False)
    importance_df.to_csv('rf_feature_importance.csv', index=False)
    
    print(f"\n📁 Resultados salvos em:")
    print(f"   - feature_analysis_results.csv")
    print(f"   - rf_feature_importance.csv")
    print(f"   - feature_importance_rf.png")

if __name__ == "__main__":
    analyze_feature_importance()