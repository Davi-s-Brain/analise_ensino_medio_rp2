import os
import pandas as pd
from src.data.loader import DataLoader
from src.models.mlp_model import MLPModel
from src.models.random_forest_model import RandomForestModel
from src.visualization.plots import ModelVisualizer

def main():
    """
    Função principal que executa o pipeline completo de análise:
    1. Carregamento e preparação dos dados
    2. Treinamento e avaliação do modelo MLP
    3. Treinamento e avaliação do modelo Random Forest
    4. Visualização dos resultados de ambos os modelos
    """
    
    ###########################################
    # 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS #
    ###########################################
    
    # Inicializa o carregador de dados com o arquivo de transição
    data_inse = DataLoader('data/TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx')
    
    if os.path.exists('data/data_combined.csv'):
        print("Carregando dados combinados do arquivo CSV existente...")
        inse_with_inep = pd.read_csv('data/data_combined.csv', encoding='utf-8-sig')
    else:
        # Combina diferentes fontes de dados em um único DataFrame
        print("Combinando dados de várias fontes...")
        inse_with_inep = data_inse.combine_data(
            data_inse.create_transicao_table(),
            data_inse.create_inse_table(),
            data_inse.create_basic_education_table(),
            data_inse.create_afd_table(),
            data_inse.create_ied_table(),
            data_inse.create_ideb_table(),
            data_inse.create_atu_table(),
            data_inse.create_had_table(),
            data_inse.create_dsu_table(),
            data_inse.create_ird_table(),
            data_inse.create_tdi_table(),
            data_inse.create_rmd_table(),
            data_inse.create_tnr_table(),
            data_inse.create_rendimento_table(),
            data_inse.create_idh_table(file_path='data/mundo_onu_adh.csv'),
            data_inse.create_raca_table('data/POP_COR_SEXO.zip', 'data/RACA_MUNICIPIOS_LIMPADO.csv'),
            data_inse.create_bolsa_familia_table('data/bolsa_familia_2021.csv', 'data/BOLSA_FAMILIA_LIMPADO.csv')
        )
    
    # Prepara os dados dividindo em conjuntos de treino e teste
    X_train_scaled, X_test_scaled, y_train, y_test = data_inse.prepare_data(inse_with_inep)

    ##############################
    #       2. MODELO MLP        #
    ##############################
    
    print("\n=== Análise do Modelo MLP ===")
    
    print("Convertendo labels de texto para inteiros (0, 1, 2, 3) para o MLP...")
    y_train_int = y_train.astype('category').cat.codes
    y_test_int = y_test.astype('category').cat.codes
    # Treinamento do modelo
    model = MLPModel(input_dim=X_train_scaled.shape[1])
    history = model.train(X_train_scaled, y_train_int)
    
    # Avaliação do modelo
    predictions = model.predict(X_test_scaled)
    mlp_metrics = model.evaluate(X_test_scaled, y_test_int)
    
    # Visualização dos resultados
    visualizer = ModelVisualizer()
    visualizer.plot_learning_curve(history)  # Curva de aprendizado
    # r2 = visualizer.plot_predictions_vs_real(y_test_int, predictions)  # Predições vs valores reais
    # visualizer.plot_error_distribution(y_test_int, predictions)  # Distribuição dos erros
    visualizer.plot_confusion_matrix(y_test_int, predictions)  # Matriz de confusão
    
    # Exibição das métricas
    metrics = {
        'Acurácia': mlp_metrics['accuracy'],
        'F1-Score (Macro)': mlp_metrics['f1_score_macro'],
        'Loss': mlp_metrics['loss']
    }

    print("\nMétricas do Modelo MLP:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    visualizer.plot_metrics(metrics)
    
    print("Gerando Curva ROC para o MLP...")
    # Pega as probabilidades
    mlp_probabilities = model.predict_proba(X_test_scaled)
    # Plota
    visualizer.plot_roc_curve(
        y_test_int,             # Os labels verdadeiros (0, 1, 2, 3)
        mlp_probabilities,      # As probabilidades
        model.class_labels,     # A lista [0, 1, 2, 3]
        model_name='MLP'
    )

    #############################
    # 3. MODELO RANDOM FOREST   #
    #############################
    
    print("\n=== Análise do Random Forest ===")
    
    # Treinamento do modelo
    rf_model = RandomForestModel()
    rf_model.train(X_train_scaled, y_train)
    
    # Avaliação do modelo
    rf_predictions = rf_model.predict(X_test_scaled)
    rf_metrics = rf_model.evaluate(X_test_scaled, y_test)
    
    # Visualização dos resultados
    rf_visualizer = ModelVisualizer()
    
    # --- CORREÇÃO ---
    # Essas duas funções são para REGRESSÃO e não funcionam com Classificação.
    # rf_visualizer.plot_predictions_vs_real_rf(y_test, rf_predictions)
    # rf_visualizer.plot_error_distribution_rf(y_test, rf_predictions)
    
    # O gráfico CORRETO de "Predição vs Real" para Classificação é a Matriz de Confusão:
    print("Gerando Matriz de Confusão para o Random Forest...")
    # O 'threshold' não faz mais sentido, pois temos 4 classes
    rf_visualizer.plot_confusion_matrix_rf(y_test, rf_predictions)
    
    # Exibição das métricas
    rf_metrics_dict = {
        'Acurácia': rf_metrics['accuracy'],
        'F1-Score (Macro)': rf_metrics['f1_score_macro']
    }

    rf_visualizer.plot_metrics_rf(rf_metrics_dict)
    
    print("Gerando Curva ROC para o Random Forest...")
    # Pega as probabilidades
    rf_probabilities = rf_model.predict_proba(X_test_scaled)
    # Plota
    rf_visualizer.plot_roc_curve(
        y_test,                 # Os labels verdadeiros (texto)
        rf_probabilities,       # As probabilidades
        rf_model.class_labels,  # A lista de textos (ex: 'Alta Evasão'...)
        model_name='Random Forest'
    )
    
    # ...
    print("\n--- 10 Features Mais Importantes (Random Forest) ---")
    try:
        # Pega os nomes das features que foram usadas (armazenadas no loader)
        feature_names = data_inse.feature_names
        
        # Chama o novo método que criamos
        importances_df = rf_model.get_feature_importances(feature_names)
        
        # Imprime as 10 mais importantes
        print(importances_df.head(10).to_string())
        
    except Exception as e:
        print(f"Erro ao obter feature importances: {e}")

if __name__ == "__main__":
    main()