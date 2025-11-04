# RELATÓRIO DE ANÁLISE E RECOMENDAÇÕES PARA MELHORIA DO R²

## SITUAÇÃO ATUAL

### Performance dos Modelos
- **Random Forest**: R² = 0.44 (configuração original simplificada)
- **MLP**: R² = 0.42 (configuração original simplificada)

### Descobertas Importantes

#### 1. **OVERFITTING DETECTADO**
- O pipeline com todas as features (incluindo NO_MUNICIPIO) resulta em R² = 0.95-0.98
- Isso indica **vazamento de dados** ou **overfitting severo**
- A variável `NO_MUNICIPIO` está causando memorização em vez de generalização

#### 2. **ANÁLISE DAS FEATURES**
- **159 variáveis** no dataset original
- **212 pares** com correlação > 0.8 (multicolinearidade alta)
- **148 variáveis** com missing values
- **35 features** removidas por correlação > 0.95

#### 3. **VARIÁVEIS MAIS IMPORTANTES**
Top 10 features por consenso dos métodos:
1. `RISCO_PEDAGOGICO_TDI_ATU` (importância: 26.2%)
2. `RISCO_INFRA_TDI_NET`
3. `VL_OBSERVADO_2021` (IDEB)
4. `PC_NIVEL_5`, `PC_NIVEL_7` (INSE)
5. `MEDIA_INSE`
6. `PC_NIVEL_6`, `PC_NIVEL_3`
7. `NO_UF_PA` (fator geográfico)
8. `RISCO_GOVERNANCA_IDH`

## RECOMENDAÇÕES PRINCIPAIS

### 🎯 **RECOMENDAÇÃO 1: FEATURE SELECTION INTELIGENTE**

**Implementar pipeline em 3 estágios:**

```python
# Estágio 1: Remoção de vazamentos e features problemáticas
features_to_remove = [
    'NO_MUNICIPIO',  # Causa overfitting
    'tx_promocao_EM', 'tx_repetencia_EM',  # Muito correlacionadas com target
    'tx_aprovacao_EM', 'tx_abandono_EM'   # Vazamento de dados
]

# Estágio 2: Feature selection por importância
consensus_features = [
    'RISCO_PEDAGOGICO_TDI_ATU',
    'RISCO_INFRA_TDI_NET', 
    'VL_OBSERVADO_2021',
    'MEDIA_INSE',
    'PC_NIVEL_5', 'PC_NIVEL_7', 'PC_NIVEL_6', 'PC_NIVEL_3',
    'RISCO_GOVERNANCA_IDH',
    'ADH_IDHM', 'ADH_EXPECTATIVA_ANOS_ESTUDO',
    # Adicionar features geográficas importantes
    'NO_UF_PA', 'NO_REGIAO_Sudeste'
]

# Estágio 3: Feature engineering
# Criar ratios e indicadores compostos
```

### 🎯 **RECOMENDAÇÃO 2: FEATURE ENGINEERING**

**Criar features de interação relevantes:**

```python
# Indicadores compostos
df['RISCO_COMPOSTO'] = (df['RISCO_PEDAGOGICO_TDI_ATU'] + 
                        df['RISCO_INFRA_TDI_NET'] + 
                        df['RISCO_GOVERNANCA_IDH']) / 3

# Ratios socioeconômicos
df['INSE_QUALIDADE_RATIO'] = df['MEDIA_INSE'] / (df['VL_OBSERVADO_2021'] + 1e-8)

# Indicador de vulnerabilidade
df['VULNERABILIDADE_SOCIAL'] = (df['ADH_PROP_POBREZA_EXTREMA'] + 
                                df['ADH_TX_ANALFABETISMO_25_MAIS']) / 2

# Peso INSE por níveis
weights = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
df['INSE_WEIGHTED'] = sum(df[f'PC_NIVEL_{i}'] * weights[i] for i in range(1, 8))
```

### 🎯 **RECOMENDAÇÃO 3: OTIMIZAÇÃO DOS MODELOS**

#### **Random Forest Otimizado:**
```python
# Usar GridSearchCV para otimização
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3]
}

# Treinar com validação cruzada
rf_model.train(X_train, y_train, tune=True, param_grid=param_grid, cv=5)
```

#### **MLP Melhorado:**
```python
# Arquitetura otimizada
model = Sequential([
    Dense(128, activation='relu', input_dim=input_dim),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

# Otimizador e learning rate
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='huber',  # Mais robusto a outliers
    metrics=['mae', 'mse']
)
```

### 🎯 **RECOMENDAÇÃO 4: TRATAMENTO DE DADOS**

#### **Outliers:**
```python
# Método mais conservador para outliers
def remove_outliers_conservative(df, target, factor=2.5):
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[target] >= Q1 - factor*IQR) & 
              (df[target] <= Q3 + factor*IQR)]
```

#### **Missing Values:**
```python
# Estratégia híbrida de imputação
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Para features numéricas importantes
iterative_imputer = IterativeImputer(random_state=42)

# Para features categóricas
# Usar moda ou categoria "Desconhecido"
```

### 🎯 **RECOMENDAÇÃO 5: VALIDAÇÃO E REGULARIZAÇÃO**

#### **Validação Cruzada Estratificada:**
```python
from sklearn.model_selection import StratifiedKFold

# Estratificar por quintis do target
y_binned = pd.qcut(y, q=5, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Avaliar com validação cruzada
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='r2')
```

#### **Regularização Adequada:**
```python
# Para Random Forest
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,  # Evita overfitting
    min_samples_leaf=4,    # Regularização
    max_features='sqrt',   # Reduz variância
    random_state=42
)

# Para MLP
# L1/L2 regularization já implementada
# Early stopping com paciência adequada
```

## EXPECTATIVAS DE MELHORIA

### **Meta Realista de R²:**
- **Random Forest**: 0.55 - 0.65 (melhoria de 25-48%)
- **MLP**: 0.50 - 0.60 (melhoria de 19-43%)

### **Fatores Limitantes:**
1. **Natureza do problema**: Evasão escolar é influenciada por muitos fatores não capturados
2. **Qualidade dos dados**: 39% de missing values em variáveis do IDEB
3. **Variabilidade regional**: Brasil tem grande diversidade socioeconômica

## IMPLEMENTAÇÃO PRIORITÁRIA

### **Fase 1 (Impacto Alto, Esforço Baixo):**
1. Remover `NO_MUNICIPIO` e features de vazamento
2. Implementar feature selection por consenso
3. Otimizar hiperparâmetros do Random Forest

### **Fase 2 (Impacto Médio, Esforço Médio):**
1. Criar features de interação
2. Implementar imputação iterativa
3. Ajustar arquitetura do MLP

### **Fase 3 (Impacto Alto, Esforço Alto):**
1. Implementar ensemble de modelos
2. Validação cruzada estratificada
3. Análise de feature importance dinâmica

## MONITORAMENTO

### **Métricas de Acompanhamento:**
- **R² ajustado** (penaliza features em excesso)
- **MAE** (interpretabilidade direta)
- **Validação cruzada** (robustez)
- **Feature importance stability** (consistência)

### **Alertas:**
- R² > 0.80 → Investigar overfitting
- MAE < 1.0 → Verificar vazamento de dados
- Diferença treino/teste > 0.15 → Regularizar mais

---

**💡 INSIGHT PRINCIPAL**: O problema não é falta de dados, mas excesso de ruído e vazamento. Focar em **qualidade sobre quantidade** de features trará melhores resultados que adicionar mais variáveis.