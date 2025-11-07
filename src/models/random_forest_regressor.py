import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class RandomForestRegressorModel:
    def __init__(self, random_state=42):
        self.model = RandomForestRegressor(random_state=random_state)
        self.cv_results = None

    def train(self, X_train, y_train, tune=False, param_grid=None, cv=3):
        if not tune:
            print("Iniciando treino com os parâmetros OTIMIZADOS do GridSearch (Regressor)...")
            # Aqui estão os parâmetros otimizados do seu último teste de Regressão
            # Se quiser, pode mudar para o n_estimators=100 simples
            self.model.set_params(
                n_estimators=300,
                criterion='squared_error',
                max_depth=30,
                min_samples_leaf=1,
                min_samples_split=2,
                random_state=42
            ) 
            self.model.fit(X_train, y_train)
            print("Treino otimizado concluído.")
        else:
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 30, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 4]
                }

            print("Iniciando otimização (GridSearchCV) para Regressor...")

            grid_search = GridSearchCV(
                estimator=self.model, param_grid=param_grid, cv=cv, 
                n_jobs=-1, verbose=2, scoring='r2'
            )

            grid_search.fit(X_train, y_train)
            print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

            self.model = grid_search.best_estimator_

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        
        print(f"\n--- Métricas de Avaliação (Random Forest Regressor) ---")
        print(f"Coeficiente R²: {r2:.4f}")
        print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        
        return {"r2": r2, "mae": mae, "mse": mse}

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importances(self, feature_names):
        importances = self.model.feature_importances_
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

        return importances_df