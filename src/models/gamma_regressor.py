import pandas as pd
from sklearn.linear_model import GammaRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class GammaRegressorModel:
    def __init__(self):
        # O 'alpha=1' é a regularização (L2). 
        # 'max_iter=1000' para garantir a convergência
        self.model = GammaRegressor(alpha=1, max_iter=1000)
        self.cv_results = None

    def train(self, X_train, y_train, tune=False, cv=3):
        """
        Treina o GammaRegressor.
        IMPORTANTE: X e y DEVEM ser > 0.
        """
        print("Iniciando treino do Gamma Regressor...")
        
        if tune:
            print("Iniciando otimização (GridSearchCV) para GammaRegressor...")
            # 'alpha' é o parâmetro de regularização (força)
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1,
                verbose=2,
                scoring='r2'
            )
            grid_search.fit(X_train, y_train)
            print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X_train, y_train)
            
        print("Treino concluído.")

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        
        print(f"\n--- Métricas de Avaliação (Gamma Regressor) ---")
        print(f"Coeficiente R²: {r2:.4f}")
        print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        
        return {"r2": r2, "mae": mae, "mse": mse}

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importances(self, feature_names):
        """
        Retorna os coeficientes do modelo como 'importância'.
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Modelo ainda não treinado.")
        
        # Modelos lineares (como Gamma) usam 'coef_'
        importances = self.model.coef_
        
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        importances_df['abs_importance'] = abs(importances_df['importance'])
        importances_df = importances_df.sort_values(
            by='abs_importance', ascending=False
        ).drop(columns=['abs_importance'])
        
        return importances_df