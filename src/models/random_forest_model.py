import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV # Importado para otimização

class RandomForestModel:
    def __init__(self, random_state=42):
        """
        Inicializa o modelo base. 
        Os hiperparâmetros serão definidos no treino.
        """
        # Modelo base simples que será usado para o fit ou para o GridSearchCV
        self.model = RandomForestRegressor(random_state=random_state)
        self.cv_results = None  # Para armazenar resultados do GridSearch

    def train(self, X_train, y_train, tune=False, param_grid=None, cv=3):
        """
        Treina o modelo.

        Args:
            X_train, y_train: Dados de treino.
            tune (bool): Se True, executa o GridSearchCV para otimizar 
                         hiperparâmetros. Se False (padrão), executa um 
                         treino simples.
            param_grid (dict): Opcional. Grid de parâmetros para o GridSearchCV.
            cv (int): Número de folds para cross-validation na otimização.
        """
        if not tune:
            # --- Treino Simples (Rápido) ---
            print("Iniciando treino simples do Random Forest (n_estimators=100)...")
            # Define parâmetros padrão para o treino rápido
            self.model.set_params(n_estimators=100) 
            self.model.fit(X_train, y_train)
            print("Treino simples concluído.")
        
        else:
            # --- Treino Otimizado (Lento) ---
            if param_grid is None:
                # Um grid de parâmetros padrão sensato para testar
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 4]
                }
            
            print(f"Iniciando otimização de hiperparâmetros (GridSearchCV) com cv={cv}...")
            print(f"Grid de busca: {param_grid}")

            # O estimador base é o self.model (que já tem o random_state)
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1,  # Usa todos os processadores
                verbose=2,
                scoring='r2' # Otimiza para o R², que é nossa métrica principal
            )
            
            grid_search.fit(X_train, y_train)
            
            print("\nOtimização concluída.")
            print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
            print(f"Melhor score (R²) na validação cruzada: {grid_search.best_score_:.4f}")
            
            # ATUALIZA o modelo da classe para o melhor modelo encontrado!
            self.model = grid_search.best_estimator_
            self.cv_results = grid_search.cv_results_

    def evaluate(self, X_test, y_test):
        """Avalia o modelo treinado no conjunto de teste."""
        predictions = self.model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        
        # O print foi movido para o main.py, mas podemos manter aqui para log
        print(f"--- Métricas de Avaliação (Random Forest) ---")
        print(f"Coeficiente R²: {r2:.4f}")
        print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        
        return {"r2": r2, "mae": mae, "mse": mse}

    def predict(self, X):
        """Realiza predições com o modelo treinado."""
        return self.model.predict(X)

    def get_feature_importances(self, feature_names):
        """
        Retorna um DataFrame com a importância de cada feature.
        Deve ser chamado APÓS o train().

        Args:
            feature_names (list): Lista de nomes das colunas (features) 
                                  na mesma ordem que foram usadas no treino.

        Returns:
            pd.DataFrame: DataFrame com colunas 'feature' e 'importance',
                          ordenado pela importância.
        """
        if not hasattr(self.model, 'feature_importances_'):
            # Erro se 'train' não foi chamado
            raise ValueError("Modelo ainda não treinado. Chame o método train() primeiro.")

        importances = self.model.feature_importances_
        
        if len(importances) != len(feature_names):
            # Erro se a lista de nomes tiver o tamanho errado
            raise ValueError(
                f"Incompatibilidade de features: O modelo tem {len(importances)} importâncias, "
                f"mas {len(feature_names)} nomes foram fornecidos."
            )

        # Cria o DataFrame
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Ordena pela importância
        importances_df = importances_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        
        return importances_df