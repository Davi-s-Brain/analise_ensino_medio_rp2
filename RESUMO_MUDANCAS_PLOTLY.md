# Resumo das Mudanças - Visualizações com Plotly

## O que foi solicitado

O usuário pediu para:
1. **Refazer os gráficos usando Plotly** (biblioteca interativa)
2. **NÃO fazer gráficos comparando com/sem vazamento** - apenas dados limpos
3. **Remover variáveis de baixa importância**, especialmente de localização

## O que foi implementado

### 1. Biblioteca Plotly Adicionada
- ✅ Adicionado `plotly==5.18.0` ao `requirements.txt`
- ✅ Adicionado `kaleido==0.2.1` para export de imagens estáticas

### 2. Script Novo: `plotly_visualizations.py`
Um script completo que gera visualizações interativas usando Plotly, com as seguintes funcionalidades:

#### Filtros Aplicados
- **Removidas 34 variáveis de localização**: Todas as variáveis começando com `NO_UF_*` e `NO_REGIAO_*`
- **Removidas 35 variáveis de baixa importância**: Variáveis com importância < 1%
- **Total: 30 features selecionadas** (de 68 originais)

#### Remoção de Vazamento de Dados
O script remove automaticamente as 8 variáveis que causam vazamento:
1. `tx_promocao_EM`
2. `tx_repetencia_EM`
3. `tx_evasao_1_ano_EM`
4. `tx_evasao_2_ano_EM`
5. `tx_evasao_3_ano_EM`
6. `tx_aprovacao_EM`
7. `tx_abandono_EM`
8. `tx_migracao_eja_EM`

### 3. Visualizações Geradas

Todas as visualizações são salvas em `plotly_output/`:

#### 3.1. Feature Importance (`feature_importance.html/png`)
- Top 20 features mais importantes
- **SEM variáveis de localização**
- **SEM variáveis de baixa importância**
- Escala de cores para destacar importância
- Valores percentuais mostrados

#### 3.2. Target Distribution (`target_distribution.html/png`)
- 4 painéis de análise:
  1. Histograma da distribuição
  2. Boxplot para detecção de outliers
  3. Distribuição acumulada
  4. Tabela de estatísticas descritivas
- Linhas de média e mediana
- Estatísticas: assimetria, curtose, quartis

#### 3.3. Model Performance (`model_performance.html/png`)
- Comparação de Random Forest vs MLP
- 3 métricas: R², MAE, MSE
- Linhas de referência para metas (R² = 0.5 e 0.6)
- Valores mostrados em cada barra

#### 3.4. Correlations (`correlations.html/png`)
- Top 15 correlações com taxa de evasão
- Cores: Verde (positiva) / Vermelho (negativa)
- **Variáveis de localização filtradas**
- Valores de correlação mostrados

#### 3.5. Dashboard Completo (`dashboard_completo.html/png`)
- Visualização consolidada em 4 painéis:
  1. Top 10 features mais importantes
  2. Distribuição da taxa de evasão
  3. Performance dos modelos (R²)
  4. Top 5 correlações
- Layout otimizado para visão geral

### 4. Formatos de Saída

Cada visualização é gerada em 2 formatos:

#### HTML (Interativo)
- Arquivos de 3.5-3.7 MB
- **Interatividade completa**: zoom, pan, hover para detalhes
- Abrir direto no navegador
- **NÃO commitados no Git** (muito grandes)

#### PNG (Estático)
- Arquivos de 60-130 KB
- Alta resolução (1200x600 a 1600x900 pixels)
- Para uso em relatórios e apresentações
- **Commitados no Git**

### 5. Documentação

Criado `plotly_output/README.md` com:
- Descrição de cada visualização
- Instruções de uso
- Lista de mudanças
- Como regenerar os gráficos

## Diferenças das Visualizações Anteriores

| Aspecto | Anterior (Matplotlib) | Novo (Plotly) |
|---------|----------------------|---------------|
| Biblioteca | Matplotlib | Plotly |
| Interatividade | Nenhuma | Total (zoom, pan, hover) |
| Comparações | Com/sem vazamento | **Apenas dados limpos** |
| Variáveis localização | Incluídas | **Removidas** |
| Variáveis baixa imp. | Incluídas | **Removidas (< 1%)** |
| Total de features | 68 | 30 |
| Formato | PNG apenas | HTML + PNG |

## Métricas Atuais dos Modelos

Com dados limpos (sem vazamento):
- **Random Forest**: R² = 0.4475, MAE = 2.90, MSE = 14.67
- **MLP**: R² = 0.4236, MAE = 2.95, MSE = 17.01

## Variáveis Mais Importantes (Top 10)

1. **RISCO_PEDAGOGICO_TDI_ATU** (26.2%) - Risco pedagógico
2. **NO_UF_PA** (4.9%) - Estado do Pará [REMOVIDA]
3. **VL_OBSERVADO_2021** (4.2%) - IDEB observado
4. **RISCO_INFRA_TDI_NET** (3.4%) - Risco de infraestrutura
5. **RACA_PERC_PRETA_PARDA** (2.9%) - Percentual raça
6. **PC_NIVEL_4** (2.8%) - INSE nível 4
7. **PC_NIVEL_3** (2.5%) - INSE nível 3
8. **RISCO_SOCIAL_TDI_PIB** (2.5%) - Risco social
9. **PC_NIVEL_1** (2.5%) - INSE nível 1
10. **PC_NIVEL_5** (2.3%) - INSE nível 5

**Nota**: Variável NO_UF_PA foi removida nas novas visualizações por ser variável de localização.

## Como Executar

```bash
python3 plotly_visualizations.py
```

Requisitos:
- Python 3.x
- pandas
- plotly
- scikit-learn
- numpy

## Arquivos Criados/Modificados

### Novos Arquivos
- `plotly_visualizations.py` - Script principal
- `plotly_output/README.md` - Documentação
- `plotly_output/*.png` - 5 imagens PNG
- `plotly_output/*.html` - 5 arquivos HTML (não commitados)

### Arquivos Modificados
- `requirements.txt` - Adicionado plotly e kaleido
- `.gitignore` - Adicionado `plotly_output/*.html`

## Próximos Passos Sugeridos

1. Abrir os arquivos HTML no navegador para ver interatividade
2. Usar os arquivos PNG em relatórios
3. Considerar ajustes baseados nas features mais importantes
4. Focar otimização nos indicadores de risco (TDI)
5. Considerar feature engineering com INSE

## Observações Técnicas

- Script funciona sem tensorflow instalado (não precisa de pipeline_clean completo)
- Remoção de vazamento é feita internamente
- Features limpas são definidas diretamente no script
- Exportação PNG funciona mesmo sem kaleido (usa backend padrão do Plotly)
