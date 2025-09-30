import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix

class ModelVisualizer:
    def plot_predictions_vs_real_rf(self, y_test, predictions):
        r2 = r2_score(y_test, predictions)
        print(f"R² Score (Acurácia) RF: {r2:.4f}")
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.title('Valores Reais vs. Predições (Random Forest)')
        plt.xlabel('Valores Reais (Taxa de Evasão)')
        plt.ylabel('Predições RF')
        plt.savefig('grafico_real_vs_predito_rf.png')
        plt.show()
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
        plt.show()

    def plot_metrics_rf(self, metrics):
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Métricas de Desempenho (Random Forest)')
        plt.ylabel('Valor')
        plt.ylim(0, 1)
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        plt.savefig('grafico_metricas_rf.png')
        plt.show()

    def plot_confusion_matrix_rf(self, y_test, predictions, threshold=0.5):
        y_test_cat = (y_test > threshold).astype(int)
        pred_cat = (predictions > threshold).astype(int)
        cm = confusion_matrix(y_test_cat, pred_cat)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Baixa', 'Alta'],
                    yticklabels=['Baixa', 'Alta'])
        plt.title('Matriz de Confusão (Random Forest)')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.savefig('grafico_matriz_confusao_rf.png')
        plt.show()
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
        plt.show()

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
        plt.show()
        
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
        plt.show()

    def plot_metrics(self, metrics):
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Métricas de Desempenho do Modelo')
        plt.ylabel('Valor')
        plt.ylim(0, 1)
        
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.savefig('grafico_metricas.png')
        plt.show()
        
    def plot_confusion_matrix(self, y_test, predictions, threshold=0.5):
      y_test_cat = (y_test > threshold).astype(int)
      pred_cat = (predictions > threshold).astype(int)
      
      cm = confusion_matrix(y_test_cat, pred_cat)
      
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=['Baixa', 'Alta'],
                  yticklabels=['Baixa', 'Alta'])
      plt.title('Matriz de Confusão\n(Evasão: Baixa vs Alta)')
      plt.xlabel('Predito')
      plt.ylabel('Real')
      plt.savefig('grafico_matriz_confusao.png')
      plt.show()