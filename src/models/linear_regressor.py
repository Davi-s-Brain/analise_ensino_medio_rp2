# Nome do novo arquivo: src/models/linear_regressor.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class LinearRegressorModel:
    def __init__(self, random_state=None):
        """
        Inicializa o modelo de Regressão Linear (Baseline).
        n_jobs=-1 usa todos os processadores disponíveis.
        """
        self.model = LinearRegression(n_jobs=-1)
        
    def train(self, X_train, y_train, tune=False):
        """
        Treina o modelo de Regressão Linear.
        'tune' não tem efeito, pois este é um baseline simples.
        """
        print("Iniciando treino do Linear Regressor (Baseline)...")
        if tune:
            print("Aviso: 'tune=True' não se aplica ao LinearRegressor. Usando default.")
            
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo e retorna as métricas de regressão.
        """
        predictions = self.model.predict(X_test)
        
        predictions[predictions < 0] = 0 
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        
        print(f"\n--- Métricas de Avaliação (Linear Regressor - Baseline) ---")
        print(f"Coeficiente R²: {r2:.4f}")
        print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        
        return {"r2": r2, "mae": mae, "mse": mse}

    def predict(self, X):
        """Realiza predições com o modelo treinado."""
        predictions = self.model.predict(X)
        predictions[predictions < 0] = 0
        return predictions

    def get_feature_importances(self, feature_names):
        """
        Retorna os coeficientes (coef_) do modelo como 'importância'.
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Modelo ainda não treinado. Chame o método train() primeiro.")
        
        importances = self.model.coef_
        
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Para coeficientes, o 'abs_importance' (valor absoluto) é o melhor
        # para classificar o que é "mais importante" (seja positivo ou negativo).
        importances_df['abs_importance'] = abs(importances_df['importance'])
        
        importances_df = importances_df.sort_values(
            by='abs_importance', ascending=False
        ).drop(columns=['abs_importance'])
        
        return importances_df