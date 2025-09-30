from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.history = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        return {"r2": r2, "mae": mae, "mse": mse}

    def predict(self, X):
        return self.model.predict(X)

    def plot_predictions_vs_real(self, y_test, predictions):
        r2 = r2_score(y_test, predictions)
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.title('Valores Reais vs. Predições (Random Forest)')
        plt.xlabel('Valores Reais')
        plt.ylabel('Predições')
        plt.savefig('grafico_real_vs_predito_rf.png')
        plt.show()
        return r2

    def plot_error_distribution(self, y_test, predictions):
        errors = y_test - predictions
        plt.figure(figsize=(12, 6))
        sns.histplot(errors, kde=True, bins=30)
        plt.title('Distribuição dos Erros de Previsão (Random Forest)')
        plt.xlabel('Erro (Real - Predito)')
        plt.ylabel('Frequência')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.savefig('grafico_distribuicao_erros_rf.png')
        plt.show()

    def plot_metrics(self, metrics):
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Métricas de Desempenho (Random Forest)')
        plt.ylabel('Valor')
        plt.ylim(0, 1)
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        plt.savefig('grafico_metricas_rf.png')
        plt.show()

    def plot_confusion_matrix(self, y_test, predictions, threshold=0.5):
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
