import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    f1_score
)

class MLPModel:
    def __init__(self, input_dim):
        self.model = self._build_model(input_dim)
        self.history = None
        self.class_labels = None

    def _build_model(self, input_dim):
        model = Sequential([
            Dense(64, activation='selu', input_dim=input_dim, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='selu'),
            BatchNormalization(),
            Dense(16, activation='selu'),
            Dense(3, activation='softmax') # 4 classes de saída
        ])
        
        model.compile(
            optimizer='adam',
            # 1. Loss para classificação de inteiros (0, 1, 2)
            loss='sparse_categorical_crossentropy', 
            # 2. Métrica de classificação
            metrics=['accuracy']
        )
        
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        """
        Treina o modelo.
        IMPORTANTE: y_train deve ser de inteiros (0, 1, 2), não texto.
        """
        
        # Armazena as classes (ex: [0, 1, 2])
        self.class_labels = sorted(np.unique(y_train))
        print(f"Treinando classificador MLP para as classes: {self.class_labels}")

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[early_stopping]
        )
        
        return self.history

    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo de classificação e retorna um dicionário de métricas.
        IMPORTANTE: y_test deve ser de inteiros (0, 1, 2), não texto.
        """
        
        # 1. Obter 'loss' e 'accuracy' brutos do Keras
        try:
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        except Exception as e:
            print(f"Erro na avaliação do Keras (verifique os labels): {e}")
            loss, accuracy = -1, -1

        # 2. Obter predições (probabilidades)
        predictions_proba = self.model.predict(X_test, verbose=0)
        
        # 3. Converter probabilidades para a classe final (0, 1, 2, ou 3)
        predictions = np.argmax(predictions_proba, axis=1)
        
        # 4. Calcular outras métricas
        f1_macro = f1_score(y_test, predictions, average='macro', labels=self.class_labels, zero_division=0)
        
        print(f"\n--- Métricas de Avaliação (MLP Classification) ---")
        print(f"Loss (sparse_categorical_crossentropy): {loss:.4f}")
        print(f"Acurácia Geral: {accuracy:.4f}")
        print(f"F1-Score (Macro Avg): {f1_macro:.4f}")
        
        # 5. Imprimir relatório detalhado
        print("\nRelatório de Classificação Detalhado:")
        try:
            report = classification_report(y_test, predictions, labels=self.class_labels, zero_division=0)
            print(report)
        except Exception as e:
            print(f"Erro ao gerar relatório de classificação: {e}")
            report = "N/A"
        
        # Retorna o dicionário para o main.py
        return {
            "loss": loss,
            "accuracy": accuracy,
            "f1_score_macro": f1_macro,
            "report_str": report
        }

    def predict(self, X_test):
        """Retorna as classes preditas (0, 1, 2)."""
        predictions_proba = self.model.predict(X_test, verbose=0)
        return np.argmax(predictions_proba, axis=1)
    
    def predict_proba(self, X_test):
        """Retorna as probabilidades de predição."""
        # No Keras, .predict() já retorna as probabilidades da camada softmax
        return self.model.predict(X_test, verbose=0)