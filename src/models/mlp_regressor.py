import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

class MLPRegressorModel:
    def __init__(self, input_dim):
        self.model = self._build_model(input_dim)
        self.history = None

    def _build_model(self, input_dim):
        model = Sequential([
            Dense(64, activation='selu', input_dim=input_dim, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='selu'),
            BatchNormalization(),
            Dense(16, activation='selu'),
            Dense(1) # <-- MUDANÇA: Saída única para regressão
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error', # <-- MUDANÇA
            metrics=['mean_absolute_error'] # <-- MUDANÇA
        )
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=5, mode='min', restore_best_weights=True
        )
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_split=validation_split,
            verbose=0, callbacks=[early_stopping]
        )
        return self.history

    def evaluate(self, X_test, y_test):
        # 'loss' será MSE, 'mae' será MAE
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        predictions = self.model.predict(X_test, verbose=0)
        
        # Calcular R² manualmente
        r2 = r2_score(y_test, predictions)
        
        print(f"\n--- Métricas de Avaliação (MLP Regressor) ---")
        print(f"Coeficiente R²: {r2:.4f}")
        print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
        print(f"Erro Quadrático Médio (MSE): {loss:.4f}")
        
        return {"r2": r2, "mae": mae, "mse": loss}

    def predict(self, X_test):
        return self.model.predict(X_test, verbose=0)