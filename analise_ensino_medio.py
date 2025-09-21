import pandas as pd

import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


df = (pd.read_excel(
        'TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx',
        skiprows=8,
        usecols=[1,2,4,5,6,51,52,53,54]
    )
    .rename(columns={'1_CAT3_CATMED': 'tx_evasao_total_EM',
                     '1_CAT3_CATMED_01': 'tx_evasao_1_ano_EM',
                     '1_CAT3_CATMED_02': 'tx_evasao_2_ano_EM',
                     '1_CAT3_CATMED_03': 'tx_evasao_3_ano_EM'})
)

df = (df.query("NO_LOCALIZACAO == 'Total' and NO_DEPENDENCIA == 'Pública'").drop(columns=['NO_LOCALIZACAO', 'NO_DEPENDENCIA']))

colunas_interesse = ['tx_evasao_total_EM', 'tx_evasao_1_ano_EM', 'tx_evasao_2_ano_EM', 'tx_evasao_3_ano_EM']
for col in colunas_interesse:
    df = df[(df[col] != '--') & (df[col] != '***')]

df[colunas_interesse] = df[colunas_interesse].astype(float)

df = df.dropna(subset=['tx_evasao_total_EM'])
df = df.fillna(df.mean(numeric_only=True))

print(df.columns.to_list())

X = df[['tx_evasao_1_ano_EM', 'tx_evasao_2_ano_EM', 'tx_evasao_3_ano_EM']]
y = df['tx_evasao_total_EM']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model_mlp = Sequential([
    # Camada de entrada e primeira camada oculta.
    # input_dim é o número de features que seu modelo recebe.
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),

    # Camada oculta intermediária com Dropout para evitar overfitting
    Dense(64, activation='relu'),
    Dropout(0.4), # "Desliga" 30% dos neurônios aleatoriamente durante o treino

    # Terceira camada oculta
    Dense(32, activation='relu'),

    # Camada de Saída: 1 neurônio para prever o valor da taxa de evasão
    Dense(1, activation='linear')
])

# Compilar o modelo
model_mlp.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

early_stopping_callback = EarlyStopping(
      monitor='val_loss', 
      patience=5,          
      mode='min',      
      restore_best_weights=True
  )

# Treinar o modelo
history = model_mlp.fit(
    X_train_scaled,
    y_train,
    epochs=100, # Aumente se o modelo continuar aprendendo
    batch_size=32,
    validation_split=0.1, # Usa 10% dos dados de treino para validação interna
    verbose=0, # verbose=0 para não poluir a saída durante o treino
    callbacks=[early_stopping_callback]
)

print("Treinamento do MLP concluído!")

# Avaliar o modelo
loss, mae = model_mlp.evaluate(X_test_scaled, y_test)
print(f"\nErro Absoluto Médio no Teste: {mae:.2f}")


# Para um visual mais agradável nos gráficos
sns.set_style("whitegrid")

# --- Gráfico 1: Curva de Aprendizagem (Loss) ---
# Mostra se o modelo está aprendendo bem ou sofrendo overfitting

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Perda no Treino')
plt.plot(history.history['val_loss'], label='Perda na Validação')
plt.title('Curva de Aprendizagem do Modelo MLP')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (Loss)')
plt.legend()
plt.savefig('grafico_curva_aprendizagem.png')
plt.show()


# --- Gráfico 2: Predições vs. Valores Reais ---
# Idealmente, os pontos devem formar uma linha reta a 45 graus

predictions = model_mlp.predict(X_test_scaled).flatten()


# Calculando R² (coeficiente de determinação)
r2 = r2_score(y_test, predictions)
print(f"R² Score (Acurácia): {r2:.4f}")


plt.figure(figsize=(10, 10))
sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2) # Linha de referência
plt.title('Valores Reais vs. Predições do Modelo')
plt.xlabel('Valores Reais (Taxa de Evasão)')
plt.ylabel('Predições')
plt.savefig('grafico_real_vs_predito.png')
plt.show()


# --- Gráfico 3: Distribuição dos Erros (Resíduos) ---
# Idealmente, os erros devem se concentrar em zero, como um sino

errors = y_test - predictions

plt.figure(figsize=(12, 6))
sns.histplot(errors, kde=True, bins=30)
plt.title('Distribuição dos Erros de Previsão (Resíduos)')
plt.xlabel('Erro (Real - Predito)')
plt.ylabel('Frequência')
plt.axvline(x=0, color='r', linestyle='--')
plt.savefig('grafico_distribuicao_erros.png')
plt.show()

# Criando um gráfico de métricas
metrics = {
    'MAE': mae,
    'R²': r2,
    'MSE': loss
}

plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values())
plt.title('Métricas de Desempenho do Modelo')
plt.ylabel('Valor')
plt.ylim(0, 1)  # Ajuste este limite conforme necessário

# Adicionar valores sobre as barras
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

plt.savefig('grafico_metricas.png')
plt.show()