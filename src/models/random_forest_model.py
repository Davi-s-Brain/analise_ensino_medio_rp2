import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    f1_score
)

class RandomForestModel:
    def __init__(self, random_state=42):
        """
        Inicializa o modelo de CLASSIFICAÇÃO.
        """
        self.model = RandomForestClassifier(random_state=random_state)
        self.cv_results = None
        self.class_labels = None

    def train(self, X_train, y_train, tune=False, param_grid=None, cv=3):
        """
        Treina o modelo de classificação.
        """
        
        # Armazena as classes (ex: ['Alta Evasão', 'Baixa Evasão', ...])
        self.class_labels = sorted(y_train.unique())
        print(f"Treinando classificador para as classes: {self.class_labels}")

        if not tune:
            print("Iniciando treino com os parâmetros OTIMIZADOS do GridSearch...")
            
            # Parâmetros encontrados: {'criterion': 'entropy', 'max_depth': None, 
            # 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
            
            self.model.set_params(
                n_estimators=200,
                criterion='entropy',
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=2,
                random_state=42
            ) 
            
            self.model.fit(X_train, y_train)
            print("Treino otimizado concluído.")
        
        else:
            if param_grid is None:
                # Grid de parâmetros padrão para classificação
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 4],
                    'criterion': ['gini', 'entropy']
                }
            
            print(f"Iniciando otimização de hiperparâmetros (GridSearchCV) com cv={cv}...")
            
            # Otimiza pela 'accuracy' (acurácia), não mais 'r2'
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1,
                verbose=2,
                scoring='accuracy' 
            )
            
            grid_search.fit(X_train, y_train)
            
            print("\nOtimização concluída.")
            print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
            print(f"Melhor score (Acurácia) na validação cruzada: {grid_search.best_score_:.4f}")
            
            self.model = grid_search.best_estimator_
            self.cv_results = grid_search.cv_results_

    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo de classificação e retorna um dicionário de métricas.
        """
        predictions = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        # 'macro' calcula a média do F1-score para cada classe, bom para desbalanceamento
        f1_macro = f1_score(y_test, predictions, average='macro', labels=self.class_labels, zero_division=0)
        
        print(f"\n--- Métricas de Avaliação (Random Forest Classification) ---")
        print(f"Acurácia Geral: {accuracy:.4f}")
        print(f"F1-Score (Macro Avg): {f1_macro:.4f}")
        
        # Imprime o relatório detalhado (Precision, Recall, F1-Score por classe)
        print("\nRelatório de Classificação Detalhado:")
        try:
            report = classification_report(y_test, predictions, labels=self.class_labels, zero_division=0)
            print(report)
        except Exception as e:
            print(f"Erro ao gerar relatório de classificação: {e}")
            report = "N/A"
        
        # Retorna o dicionário para o main.py
        return {
            "accuracy": accuracy,
            "f1_score_macro": f1_macro,
            "report_str": report
        }

    def predict(self, X):
        """Realiza predições com o modelo treinado."""
        return self.model.predict(X)

    def get_feature_importances(self, feature_names):
        """
        Retorna um DataFrame com a importância de cada feature.
        (Esta função continua igual, é válida para ambos os modelos)
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Modelo ainda não treinado. Chame o método train() primeiro.")

        importances = self.model.feature_importances_
        
        if len(importances) != len(feature_names):
            raise ValueError(
                f"Incompatibilidade de features: O modelo tem {len(importances)} importâncias, "
                f"mas {len(feature_names)} nomes foram fornecidos."
            )

        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        importances_df = importances_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        
        return importances_df
    
    def predict_proba(self, X):
        """Retorna as probabilidades de predição."""
        return self.model.predict_proba(X)