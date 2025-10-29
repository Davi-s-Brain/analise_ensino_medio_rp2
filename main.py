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
    
    # Combina diferentes fontes de dados em um único DataFrame
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
        data_inse.create_rendimento_table()
    )
    
    # Prepara os dados dividindo em conjuntos de treino e teste
    X_train_scaled, X_test_scaled, y_train, y_test = data_inse.prepare_data(inse_with_inep)

    ##############################
    #       2. MODELO MLP        #
    ##############################
    
    print("\n=== Análise do Modelo MLP ===")
    
    # Treinamento do modelo
    model = MLPModel(input_dim=X_train_scaled.shape[1])
    history = model.train(X_train_scaled, y_train)
    
    # Avaliação do modelo
    predictions = model.predict(X_test_scaled)
    loss, mae = model.evaluate(X_test_scaled, y_test)
    
    # Visualização dos resultados
    visualizer = ModelVisualizer()
    visualizer.plot_learning_curve(history)  # Curva de aprendizado
    r2 = visualizer.plot_predictions_vs_real(y_test, predictions)  # Predições vs valores reais
    visualizer.plot_error_distribution(y_test, predictions)  # Distribuição dos erros
    visualizer.plot_confusion_matrix(y_test, predictions, threshold=y_test.mean())  # Matriz de confusão
    
    # Exibição das métricas
    metrics = {
        'MAE': mae,   # Erro Médio Absoluto
        'R²': r2,     # Coeficiente de determinação
        'MSE': loss   # Erro Quadrático Médio
    }

    print("\nMétricas do Modelo MLP:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    visualizer.plot_metrics(metrics)

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
    rf_visualizer.plot_predictions_vs_real_rf(y_test, rf_predictions)  # Predições vs valores reais
    rf_visualizer.plot_error_distribution_rf(y_test, rf_predictions)   # Distribuição dos erros
    rf_visualizer.plot_confusion_matrix_rf(y_test, rf_predictions, threshold=y_test.mean())  # Matriz de confusão
    
    # Exibição das métricas
    rf_metrics_dict = {
        'MAE': rf_metrics['mae'],  # Erro Médio Absoluto
        'R²': rf_metrics['r2'],    # Coeficiente de determinação
        'MSE': rf_metrics['mse']   # Erro Quadrático Médio
    }
    rf_visualizer.plot_metrics_rf(rf_metrics_dict)

if __name__ == "__main__":
    main()