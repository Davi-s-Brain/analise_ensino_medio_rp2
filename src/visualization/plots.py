import numpy as np
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

class ModelVisualizer:
    def __init__(self):
        sns.set_style("whitegrid")

    def plot_learning_curve(self, history):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Perda no Treino')
        plt.plot(history.history['val_loss'], label='Perda na Validação')
        plt.title('Curva de Aprendizagem do Modelo MLP')
        plt.xlabel('Épocas')
        plt.ylabel('Erro Quadrático Médio (Loss)')
        plt.legend()
        plt.savefig('grafico_curva_aprendizagem.png')
        # plt.show()

    def plot_predictions_vs_real(self, y_test, predictions):
        r2 = r2_score(y_test, predictions)
        print(f"R² Score (Acurácia): {r2:.4f}")
        
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                '--r', linewidth=2)
        plt.title('Valores Reais vs. Predições do Modelo')
        plt.xlabel('Valores Reais (Taxa de Evasão)')
        plt.ylabel('Predições')
        plt.savefig('grafico_real_vs_predito.png')
        # plt.show()
        
        return r2

    def plot_error_distribution(self, y_test, predictions):
        errors = y_test - predictions
        plt.figure(figsize=(12, 6))
        sns.histplot(errors, kde=True, bins=30)
        plt.title('Distribuição dos Erros de Previsão (Resíduos)')
        plt.xlabel('Erro (Real - Predito)')
        plt.ylabel('Frequência')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.savefig('grafico_distribuicao_erros.png')
        # plt.show()

    def plot_metrics(self, metrics):
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Métricas de Desempenho do Modelo')
        plt.ylabel('Valor')
        plt.ylim(0, 1)
        
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.savefig('grafico_metricas.png')
        # plt.show()
        
    def plot_confusion_matrix(self, y_test, predictions):
        """
        Plota uma matriz de confusão MULTI-CLASSE (4x4) para o MLP.
        (Versão de Classificação, sem 'threshold')
        """
        print("Gerando Matriz de Confusão (Multi-classe) para MLP...")
        
        # Pega os nomes das classes (ex: [0, 1, 2])
        # e ordena para que a matriz fique consistente
        class_labels = sorted(np.unique(y_test))
        
        # 1. Calcula a Matriz de Confusão
        cm = confusion_matrix(y_test, predictions, labels=class_labels)
        
        # 2. Plota o Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True,     # Mostra os números dentro de cada célula
            fmt='d',        # Formato dos números (inteiro)
            cmap='Blues',   # Cor
            xticklabels=class_labels, 
            yticklabels=class_labels
        )
        
        plt.title('Matriz de Confusão - MLP (Classificação)')
        plt.ylabel('Verdadeiro (Real)')
        plt.xlabel('Predito')
        
        # Salva a figura
        filename = 'grafico_matriz_confusao.png'
        plt.savefig(filename, bbox_inches='tight')
        print(f"Matriz de confusão do MLP salva em '{filename}'")
        # plt.show()
        plt.close() # Fecha a figura para economizar memória
    
    def plot_predictions_vs_real_rf(self, y_test, predictions):
        r2 = r2_score(y_test, predictions)
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.title('Valores Reais vs. Predições (Random Forest)')
        plt.xlabel('Valores Reais (Taxa de Evasão)')
        plt.ylabel('Predições RF')
        plt.savefig('grafico_real_vs_predito_rf.png')
        # plt.show()
        return r2

    def plot_error_distribution_rf(self, y_test, predictions):
        errors = y_test - predictions
        plt.figure(figsize=(12, 6))
        sns.histplot(errors, kde=True, bins=30)
        plt.title('Distribuição dos Erros de Previsão (RF)')
        plt.xlabel('Erro (Real - Predito) RF')
        plt.ylabel('Frequência')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.savefig('grafico_distribuicao_erros_rf.png')
        # plt.show()

    def plot_metrics_rf(self, metrics):
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Métricas de Desempenho (Random Forest)')
        plt.ylabel('Valor')
        plt.ylim(0, 1)
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        plt.savefig('grafico_metricas_rf.png')
        # plt.show()

    def plot_confusion_matrix_rf(self, y_test, predictions):
        """
        Plota uma matriz de confusão MULTI-CLASSE (4x4) para o Random Forest.
        (Versão de Classificação, sem 'threshold')
        """
        print("Gerando Matriz de Confusão (Multi-classe) para RF...")
        
        # Pega os nomes das classes (ex: 'Alta Evasão', 'Baixa Evasão', ...)
        # e ordena para que a matriz fique consistente
        class_labels = sorted(y_test.unique())
        
        # 1. Calcula a Matriz de Confusão
        cm = confusion_matrix(y_test, predictions, labels=class_labels)
        
        # 2. Plota o Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True,     # Mostra os números dentro de cada célula
            fmt='d',        # Formato dos números (inteiro)
            cmap='Oranges', # Cor
            xticklabels=class_labels, 
            yticklabels=class_labels
        )
        
        plt.title('Matriz de Confusão - Random Forest (Classificação)')
        plt.ylabel('Verdadeiro (Real)')
        plt.xlabel('Predito')
        
        # Salva a figura
        filename = 'grafico_matriz_confusao_rf.png'
        plt.savefig(filename, bbox_inches='tight')
        print(f"Matriz de confusão do RF salva em '{filename}'")
        # plt.show()
        plt.close() # Fecha a figura para economizar memória
        
    def plot_roc_curve(self, y_test, y_proba, class_labels, model_name=''):
        """
        Plota as curvas ROC para um problema multi-classe (One-vs-Rest).
        
        Args:
            y_test: Os labels verdadeiros (ex: [0, 1, 2] ou ['Baixa', 'Alta', ...])
            y_proba: As probabilidades de cada classe (saída do .predict_proba())
            class_labels: A lista ordenada das classes (ex: [0, 1, 2] ou ['Alta', 'Baixa', ...])
            model_name: O nome do modelo (ex: "Random Forest")
        """
        print(f"Gerando Curva ROC (One-vs-Rest) para {model_name}...")
        
        # 1. Binarizar os labels (ex: 'Média Alta' -> [0, 0, 1, 0])
        # Isso garante que funcione tanto para texto (RF) quanto para inteiros (MLP)
        y_test_binarized = label_binarize(y_test, classes=class_labels)
        n_classes = y_test_binarized.shape[1]

        # 2. Calcular a ROC e AUC para cada classe
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 3. Plotar
        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i], 
                tpr[i], 
                color=color, 
                lw=2,
                label=f'Classe: {class_labels[i]} (AUC = {roc_auc[i]:.2f})'
            )

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chute Aleatório (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)')
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
        plt.title(f'Curva ROC (One-vs-Rest) - {model_name}')
        plt.legend(loc="lower right")
        
        filename = f'grafico_curva_roc_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, bbox_inches='tight')
        print(f"Curva ROC salva em '{filename}'")
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


    def __init__(self):
        sns.set_style("whitegrid")