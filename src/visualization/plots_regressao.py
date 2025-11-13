# Nome do novo arquivo: src/visualization/plots_regressao.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class RegressorVisualizer:
    def plot_predictions_vs_real(self, y_test, predictions, model_name=''):
        """
        Plota as predições vs. os valores reais (Gráfico de Dispersão).
        """
        plt.figure(figsize=(10, 6))
        
        # Garante que 'predictions' seja um array 1D
        if hasattr(predictions, 'flatten'):
            predictions = predictions.flatten()
            
        r2 = r2_score(y_test, predictions)
        
        plt.scatter(y_test, predictions, alpha=0.5, label=f'Predições (R² = {r2:.3f})')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfeito (y=x)')
        
        plt.xlabel('Valores Reais (Taxa de Evasão)')
        plt.ylabel('Valores Preditos')
        plt.title(f'Real vs. Predito - {model_name}')
        plt.legend()
        plt.grid(True)
        
        filename = f'grafico_real_vs_predito_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def plot_error_distribution(self, y_test, predictions, model_name=''):
        """
        Plota a distribuição dos erros (Resíduos).
        """
        if hasattr(predictions, 'flatten'):
            predictions = predictions.flatten()
            
        errors = y_test - predictions
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True, bins=30)
        plt.xlabel('Erro de Predição (Real - Predito)')
        plt.ylabel('Frequência')
        plt.title(f'Distribuição dos Erros (Resíduos) - {model_name}')
        
        filename = f'grafico_distribuicao_erros_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def plot_metrics(self, metrics_dict, model_name=''):
        """
        Plota um gráfico de barras simples para as métricas (R², MAE, MSE).
        """
        df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=['Métrica', 'Valor'])
        
        plt.figure(figsize=(8, 5))
        barplot = sns.barplot(x='Métrica', y='Valor', data=df_metrics)
        
        for p in barplot.patches:
            barplot.annotate(
                f'{p.get_height():.4f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 9), 
                textcoords='offset points'
            )
            
        plt.title(f'Métricas de Regressão - {model_name}')
        plt.ylim(0, max(metrics_dict.values()) * 1.2)
        
        filename = f'grafico_metricas_regressao_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        

    def plot_feature_importance(self, importances_df, model_name, top_n=10):
        """
        Plota um gráfico de barras horizontal com as features mais importantes.
        (Versão Corrigida para Seaborn v0.14+)
        """
        
        # 1. Pegar apenas o Top N e inverter a ordem
        df_top = importances_df.head(top_n)
        df_top = df_top.iloc[::-1]

        plt.figure(figsize=(12, 8))
        
        # --- CORREÇÃO (Seaborn Warning) ---
        # O Seaborn agora exige que 'hue' seja atribuído para usar 'palette'
        # para um gradiente na variável 'y'.
        barplot = sns.barplot(
            x='importance', 
            y='feature', 
            data=df_top, 
            palette="flare",  # O degradê que você quer
            hue='feature',      # Atribui a variável y (feature) ao hue
            legend=False      # Desliga a legenda (que é desnecessária)
        )
        # --- FIM DA CORREÇÃO ---
        
        plt.title(f'Top {top_n} Features Mais Importantes - {model_name}')
        plt.xlabel('Importância Relativa')
        plt.ylabel('Feature')
        
        # Adiciona os valores nas barras
        for p in barplot.patches:
            width = p.get_width()
            plt.text(
                width * 1.005, # Posição X (um pouco depois da barra)
                p.get_y() + p.get_height() / 2, # Posição Y (meio da barra)
                f'{width:.5f}', # O texto (formatado)
                va='center'
            )
            
        filename = f'grafico_feature_importance_{model_name.lower().replace(" ", "_")}.png'
        
        # Ajusta o layout para não cortar os nomes
        plt.tight_layout() 
        
        plt.savefig(filename)
        print(f"Gráfico de Feature Importance salvo em '{filename}'")
        plt.close()