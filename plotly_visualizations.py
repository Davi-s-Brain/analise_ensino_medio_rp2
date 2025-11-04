"""
Script para criar visualizações usando Plotly (biblioteca interativa)
Dados limpos (sem vazamento) e sem variáveis de baixa importância
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os


def filter_low_importance_features(importance_df, threshold=0.01):
    """
    Remove features de baixa importância e features de localização
    
    Args:
        importance_df: DataFrame com colunas 'feature' e 'importance'
        threshold: Limiar mínimo de importância (default: 1%)
    
    Returns:
        DataFrame filtrado
    """
    # Remove features de localização (NO_UF_*, NO_REGIAO_*)
    location_features = importance_df[
        importance_df['feature'].str.contains('NO_UF_|NO_REGIAO_', case=False, na=False)
    ].index
    
    print(f"🗑️  Removendo {len(location_features)} variáveis de localização")
    
    # Remove features com importância muito baixa
    low_importance = importance_df[importance_df['importance'] < threshold].index
    print(f"🗑️  Removendo {len(low_importance)} variáveis com importância < {threshold}")
    
    # Combina os dois conjuntos
    to_remove = set(location_features) | set(low_importance)
    
    filtered_df = importance_df.drop(index=to_remove)
    
    print(f"✅ Features restantes: {len(filtered_df)} (de {len(importance_df)} originais)")
    
    return filtered_df


def create_feature_importance_plot(importance_df, top_n=20):
    """
    Cria gráfico de importância de features usando Plotly
    """
    # Filtra features de baixa importância
    filtered_df = filter_low_importance_features(importance_df, threshold=0.01)
    
    # Pega top N features
    top_features = filtered_df.nlargest(top_n, 'importance')
    
    # Cria gráfico de barras horizontal
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_features['feature'][::-1],  # Inverte para mostrar maior no topo
        x=top_features['importance'][::-1] * 100,  # Converte para percentual
        orientation='h',
        marker=dict(
            color=top_features['importance'][::-1],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importância")
        ),
        text=[f"{val:.2f}%" for val in top_features['importance'][::-1] * 100],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importância: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Top {top_n} Features Mais Importantes<br><sub>Dados Limpos (sem vazamento e sem variáveis de localização)</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Importância (%)',
        yaxis_title='Feature',
        height=600,
        template='plotly_white',
        font=dict(size=11),
        margin=dict(l=200, r=100, t=100, b=50)
    )
    
    return fig


def create_target_distribution_plot(df, target_col='tx_evasao_total_EM'):
    """
    Cria visualização da distribuição da variável target
    """
    target_data = df[target_col].dropna()
    
    # Cria subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribuição da Taxa de Evasão',
            'Boxplot - Detecção de Outliers',
            'Distribuição Acumulada',
            'Estatísticas Descritivas'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "box"}],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Histograma
    fig.add_trace(
        go.Histogram(
            x=target_data,
            nbinsx=50,
            name='Distribuição',
            marker_color='skyblue',
            marker_line_color='darkblue',
            marker_line_width=1,
            hovertemplate='Taxa: %{x:.1f}%<br>Frequência: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Adiciona linhas de média e mediana
    mean_val = target_data.mean()
    median_val = target_data.median()
    
    fig.add_vline(
        x=mean_val, line_dash="dash", line_color="red",
        annotation_text=f"Média: {mean_val:.1f}%",
        row=1, col=1
    )
    
    fig.add_vline(
        x=median_val, line_dash="dash", line_color="green",
        annotation_text=f"Mediana: {median_val:.1f}%",
        row=1, col=1
    )
    
    # 2. Boxplot
    fig.add_trace(
        go.Box(
            y=target_data,
            name='Taxa de Evasão',
            marker_color='lightcoral',
            boxmean='sd',
            hovertemplate='Valor: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Distribuição acumulada
    sorted_data = np.sort(target_data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    
    fig.add_trace(
        go.Scatter(
            x=sorted_data,
            y=cumulative,
            mode='lines',
            name='Acumulada',
            line=dict(color='purple', width=2),
            hovertemplate='Taxa: %{x:.1f}%<br>Percentil: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Tabela de estatísticas
    stats = target_data.describe()
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Estatística', 'Valor'],
                fill_color='lightblue',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[
                    ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Q1 (25%)', 'Q3 (75%)', 'Assimetria', 'Curtose'],
                    [
                        f"{mean_val:.2f}%",
                        f"{median_val:.2f}%",
                        f"{target_data.std():.2f}%",
                        f"{target_data.min():.2f}%",
                        f"{target_data.max():.2f}%",
                        f"{target_data.quantile(0.25):.2f}%",
                        f"{target_data.quantile(0.75):.2f}%",
                        f"{target_data.skew():.3f}",
                        f"{target_data.kurtosis():.3f}"
                    ]
                ],
                fill_color='white',
                align='left',
                font=dict(size=11)
            )
        ),
        row=2, col=2
    )
    
    # Layout
    fig.update_xaxes(title_text="Taxa de Evasão (%)", row=1, col=1)
    fig.update_yaxes(title_text="Frequência", row=1, col=1)
    fig.update_yaxes(title_text="Taxa de Evasão (%)", row=1, col=2)
    fig.update_xaxes(title_text="Taxa de Evasão (%)", row=2, col=1)
    fig.update_yaxes(title_text="Percentil (%)", row=2, col=1)
    
    fig.update_layout(
        title=dict(
            text='Análise da Taxa de Evasão do Ensino Médio<br><sub>Dados corrigidos (sem vazamento)</sub>',
            x=0.5,
            xanchor='center'
        ),
        height=800,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def create_model_performance_plot(results_df):
    """
    Cria visualização do desempenho dos modelos
    """
    # Reorganiza os dados
    models = results_df.index.tolist()
    metrics = ['R²', 'MAE', 'MSE']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Coeficiente R²', 'MAE (Erro Médio Absoluto)', 'MSE (Erro Quadrático Médio)'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#2ecc71', '#3498db']  # Verde para RF, Azul para MLP
    
    # R² Score
    fig.add_trace(
        go.Bar(
            x=models,
            y=results_df['R²'],
            name='R²',
            marker_color=colors,
            text=[f"{val:.3f}" for val in results_df['R²']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Bar(
            x=models,
            y=results_df['MAE'],
            name='MAE',
            marker_color=colors,
            text=[f"{val:.2f}" for val in results_df['MAE']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>MAE: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # MSE
    fig.add_trace(
        go.Bar(
            x=models,
            y=results_df['MSE'],
            name='MSE',
            marker_color=colors,
            text=[f"{val:.2f}" for val in results_df['MSE']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>MSE: %{y:.4f}<extra></extra>'
        ),
        row=1, col=3
    )
    
    # Linha de referência para R² ideal (0.5-0.6)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, 
                  annotation_text="Meta mínima", row=1, col=1)
    fig.add_hline(y=0.6, line_dash="dash", line_color="green", opacity=0.5,
                  annotation_text="Meta ideal", row=1, col=1)
    
    fig.update_layout(
        title=dict(
            text='Desempenho dos Modelos de Previsão<br><sub>Dados limpos (sem vazamento de dados)</sub>',
            x=0.5,
            xanchor='center'
        ),
        height=500,
        showlegend=False,
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="Valor", row=1, col=1)
    fig.update_yaxes(title_text="Valor", row=1, col=2)
    fig.update_yaxes(title_text="Valor", row=1, col=3)
    
    return fig


def create_correlation_plot(corr_df, top_n=15):
    """
    Cria visualização das correlações com o target
    """
    # Filtra features de localização
    corr_filtered = corr_df[
        ~corr_df['feature'].str.contains('NO_UF_|NO_REGIAO_', case=False, na=False)
    ]
    
    top_corr = corr_filtered.nlargest(top_n, 'correlation')
    
    # Cria gráfico
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_corr['correlation_signed']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_corr['feature'][::-1],
        x=top_corr['correlation'][::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(color='black', width=1)
        ),
        text=[f"{val:.3f}" for val in top_corr['correlation'][::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Correlação: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Top {top_n} Correlações com Taxa de Evasão<br><sub>Verde = correlação positiva | Vermelho = correlação negativa</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Correlação Absoluta',
        yaxis_title='Feature',
        height=600,
        template='plotly_white',
        margin=dict(l=250, r=100, t=100, b=50)
    )
    
    return fig


def create_comprehensive_dashboard(importance_df, target_data, results_df, corr_df):
    """
    Cria um dashboard completo com todas as análises principais
    """
    # Filtra features de baixa importância
    filtered_importance = filter_low_importance_features(importance_df, threshold=0.01)
    top_10 = filtered_importance.nlargest(10, 'importance')
    
    # Filtra correlações (sem localização)
    corr_filtered = corr_df[
        ~corr_df['feature'].str.contains('NO_UF_|NO_REGIAO_', case=False, na=False)
    ]
    top_5_corr = corr_filtered.nlargest(5, 'correlation')
    
    # Cria subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Top 10 Features Mais Importantes',
            'Distribuição da Taxa de Evasão',
            'Desempenho dos Modelos (R²)',
            'Top 5 Correlações com Target'
        ),
        specs=[
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # 1. Feature Importance
    fig.add_trace(
        go.Bar(
            y=top_10['feature'][::-1],
            x=top_10['importance'][::-1] * 100,
            orientation='h',
            marker=dict(color='forestgreen'),
            text=[f"{val:.1f}%" for val in top_10['importance'][::-1] * 100],
            textposition='outside',
            name='Importância',
            showlegend=False,
            hovertemplate='<b>%{y}</b><br>Importância: %{x:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Target Distribution
    fig.add_trace(
        go.Histogram(
            x=target_data,
            nbinsx=40,
            marker_color='skyblue',
            marker_line_color='darkblue',
            marker_line_width=1,
            name='Distribuição',
            showlegend=False,
            hovertemplate='Taxa: %{x:.1f}%<br>Frequência: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Adiciona média
    mean_val = target_data.mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", row=1, col=2)
    
    # 3. Model Performance
    models = results_df.index.tolist()
    colors_models = ['#2ecc71', '#3498db']
    
    fig.add_trace(
        go.Bar(
            x=models,
            y=results_df['R²'],
            marker_color=colors_models,
            text=[f"{val:.3f}" for val in results_df['R²']],
            textposition='outside',
            name='R²',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Top Correlations
    colors_corr = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_5_corr['correlation_signed']]
    
    fig.add_trace(
        go.Bar(
            y=top_5_corr['feature'][::-1],
            x=top_5_corr['correlation'][::-1],
            orientation='h',
            marker=dict(color=colors_corr[::-1]),
            text=[f"{val:.2f}" for val in top_5_corr['correlation'][::-1]],
            textposition='outside',
            name='Correlação',
            showlegend=False,
            hovertemplate='<b>%{y}</b><br>Correlação: %{x:.4f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Layout
    fig.update_xaxes(title_text="Importância (%)", row=1, col=1)
    fig.update_xaxes(title_text="Taxa de Evasão (%)", row=1, col=2)
    fig.update_yaxes(title_text="Frequência", row=1, col=2)
    fig.update_yaxes(title_text="R² Score", row=2, col=1)
    fig.update_xaxes(title_text="Correlação", row=2, col=2)
    
    fig.update_layout(
        title=dict(
            text='Dashboard Completo - Análise de Evasão do Ensino Médio<br><sub>Dados limpos sem vazamento e sem variáveis de localização</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        height=900,
        template='plotly_white',
        font=dict(size=10)
    )
    
    return fig


def main():
    """
    Função principal para gerar todas as visualizações
    """
    print("=" * 70)
    print("GERANDO VISUALIZAÇÕES COM PLOTLY - DADOS LIMPOS")
    print("=" * 70)
    
    # 1. Carrega dados
    print("\n📊 Carregando dados...")
    
    # Resultados dos modelos
    if not os.path.exists('resultados_limpos_sem_vazamento.csv'):
        print("❌ Arquivo 'resultados_limpos_sem_vazamento.csv' não encontrado!")
        print("   Execute pipeline_clean.py primeiro.")
        return
    
    results_df = pd.read_csv('resultados_limpos_sem_vazamento.csv', index_col=0)
    print(f"✅ Resultados dos modelos carregados: {results_df.shape}")
    
    # Feature importance
    if not os.path.exists('rf_feature_importance.csv'):
        print("❌ Arquivo 'rf_feature_importance.csv' não encontrado!")
        print("   Execute pipeline_clean.py primeiro.")
        return
    
    importance_df = pd.read_csv('rf_feature_importance.csv')
    print(f"✅ Feature importance carregada: {importance_df.shape}")
    
    # Dados combinados
    if not os.path.exists('data/data_combined.csv'):
        print("❌ Arquivo 'data/data_combined.csv' não encontrado!")
        return
    
    df = pd.read_csv('data/data_combined.csv', encoding='utf-8-sig')
    print(f"✅ Dados combinados carregados: {df.shape}")
    
    # Remove vazamento de dados
    print("\n🚨 Removendo vazamento de dados...")
    
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
    
    target = 'tx_evasao_total_EM'
    target_data = df[target].dropna()
    print(f"✅ Target preparado: {len(target_data)} registros")
    
    # Calcula correlações
    print("\n📈 Calculando correlações...")
    
    # Lista de features limpas (sem vazamento)
    clean_features = [
        # Features geográficas (controladas)
        'NO_REGIAO', 'NO_UF',
        
        # INSE (Índice Socioeconômico) - VÁLIDAS
        'MEDIA_INSE', 'PC_NIVEL_1', 'PC_NIVEL_2', 'PC_NIVEL_3', 
        'PC_NIVEL_4', 'PC_NIVEL_5', 'PC_NIVEL_6', 'PC_NIVEL_7',
        
        # IDEB e Qualidade (anos anteriores) - VÁLIDAS
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
    
    numeric_cols = [col for col in clean_features if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    correlations = []
    for col in numeric_cols:
        if col != target:
            corr = df[col].corr(df[target])
            if not np.isnan(corr):
                correlations.append({
                    'feature': col,
                    'correlation': abs(corr),
                    'correlation_signed': corr
                })
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    print(f"✅ Correlações calculadas: {len(corr_df)} features")
    
    # 2. Cria visualizações
    print("\n🎨 Criando visualizações Plotly...")
    
    output_dir = 'plotly_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature Importance
    print("   1/5 - Feature Importance...")
    fig1 = create_feature_importance_plot(importance_df, top_n=20)
    fig1.write_html(f'{output_dir}/feature_importance.html')
    try:
        fig1.write_image(f'{output_dir}/feature_importance.png', width=1200, height=600)
    except Exception as e:
        print(f"      ⚠️  PNG export não disponível (instale kaleido): {e}")
    
    # Target Distribution
    print("   2/5 - Target Distribution...")
    fig2 = create_target_distribution_plot(df, target)
    fig2.write_html(f'{output_dir}/target_distribution.html')
    try:
        fig2.write_image(f'{output_dir}/target_distribution.png', width=1200, height=800)
    except Exception as e:
        print(f"      ⚠️  PNG export não disponível: {e}")
    
    # Model Performance
    print("   3/5 - Model Performance...")
    fig3 = create_model_performance_plot(results_df)
    fig3.write_html(f'{output_dir}/model_performance.html')
    try:
        fig3.write_image(f'{output_dir}/model_performance.png', width=1400, height=500)
    except Exception as e:
        print(f"      ⚠️  PNG export não disponível: {e}")
    
    # Correlations
    print("   4/5 - Correlations...")
    fig4 = create_correlation_plot(corr_df, top_n=15)
    fig4.write_html(f'{output_dir}/correlations.html')
    try:
        fig4.write_image(f'{output_dir}/correlations.png', width=1200, height=600)
    except Exception as e:
        print(f"      ⚠️  PNG export não disponível: {e}")
    
    # Comprehensive Dashboard
    print("   5/5 - Comprehensive Dashboard...")
    fig5 = create_comprehensive_dashboard(importance_df, target_data, results_df, corr_df)
    fig5.write_html(f'{output_dir}/dashboard_completo.html')
    try:
        fig5.write_image(f'{output_dir}/dashboard_completo.png', width=1600, height=900)
    except Exception as e:
        print(f"      ⚠️  PNG export não disponível: {e}")
    
    print(f"\n✅ VISUALIZAÇÕES CRIADAS COM SUCESSO!")
    print(f"\n📁 Arquivos salvos em '{output_dir}/':")
    print("   • feature_importance.html")
    print("   • target_distribution.html")
    print("   • model_performance.html")
    print("   • correlations.html")
    print("   • dashboard_completo.html")
    print("\n💡 Arquivos .html são interativos (abra no navegador)")
    print("💡 Para gerar .png, instale: pip install kaleido")
    
    print("\n" + "=" * 70)
    print("RESUMO DAS MUDANÇAS:")
    print("=" * 70)
    print("✅ Variáveis de localização (NO_UF_*, NO_REGIAO_*) removidas")
    print("✅ Variáveis com importância < 1% removidas")
    print("✅ Apenas dados limpos (sem vazamento)")
    print("✅ Visualizações interativas com Plotly")
    print("✅ Sem comparações com dados vazados")
    print("=" * 70)


if __name__ == "__main__":
    main()
