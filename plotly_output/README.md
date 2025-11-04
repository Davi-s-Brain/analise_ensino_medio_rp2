# Visualizações Interativas com Plotly

Este diretório contém visualizações interativas geradas usando a biblioteca Plotly.

## Arquivos Gerados

### 1. Feature Importance (`feature_importance.html`)
- Mostra as 20 features mais importantes para prever a taxa de evasão
- **Variáveis de localização removidas** (NO_UF_*, NO_REGIAO_*)
- **Variáveis com importância < 1% removidas**
- Apenas dados limpos (sem vazamento de dados)

### 2. Target Distribution (`target_distribution.html`)
- Análise completa da distribuição da taxa de evasão
- Inclui histograma, boxplot, distribuição acumulada e estatísticas descritivas
- Dados corrigidos (sem vazamento)

### 3. Model Performance (`model_performance.html`)
- Comparação do desempenho dos modelos Random Forest e MLP
- Métricas: R², MAE, MSE
- Resultados com dados limpos

### 4. Correlations (`correlations.html`)
- Top 15 correlações com a taxa de evasão
- Correlações positivas (verde) e negativas (vermelho)
- Variáveis de localização excluídas

### 5. Dashboard Completo (`dashboard_completo.html`)
- Visualização consolidada com:
  - Top 10 features mais importantes
  - Distribuição da taxa de evasão
  - Desempenho dos modelos (R²)
  - Top 5 correlações com target

## Como Visualizar

### Arquivos HTML (Interativos)
Os arquivos `.html` são **interativos** e podem ser abertos diretamente no navegador:
1. Abra o arquivo `.html` no seu navegador preferido
2. Use o mouse para:
   - **Zoom**: Clique e arraste para dar zoom em áreas específicas
   - **Pan**: Segure Shift e arraste para mover o gráfico
   - **Hover**: Passe o mouse sobre os elementos para ver detalhes
   - **Reset**: Clique duas vezes para resetar o zoom

### Arquivos PNG (Estáticos)
Os arquivos `.png` são imagens estáticas para uso em relatórios e apresentações.

## Mudanças em Relação às Visualizações Anteriores

✅ **Biblioteca atualizada**: Matplotlib → Plotly (interativa)
✅ **Variáveis de localização removidas**: NO_UF_*, NO_REGIAO_*
✅ **Variáveis de baixa importância removidas**: Importância < 1%
✅ **Apenas dados limpos**: Sem vazamento de dados
✅ **Sem comparações**: Removidas comparações com dados vazados
✅ **Visualizações mais limpas**: Foco nas variáveis relevantes

## Geração dos Gráficos

Para gerar novamente as visualizações, execute:

```bash
python3 plotly_visualizations.py
```

## Requisitos

- Python 3.x
- pandas
- plotly
- scikit-learn
- numpy

Para instalar as dependências:

```bash
pip install -r requirements.txt
```

## Dados Utilizados

- `resultados_limpos_sem_vazamento.csv`: Métricas dos modelos
- `rf_feature_importance.csv`: Importância das features
- `data/data_combined.csv`: Dados combinados (com remoção de vazamento)
