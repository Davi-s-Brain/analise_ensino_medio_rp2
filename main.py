from src.data.loader import DataLoader
from src.models.mlp_model import MLPModel
from src.models.random_forest_model import RandomForestModel
from src.visualization.plots import ModelVisualizer

def main():
    # Carregar e preparar dados
    data_inse = DataLoader('data/TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx')
    inse_with_inep = data_inse.combine_data(
        data_inse.load_data(),
        data_inse.create_inse_table(),
        data_inse.create_basic_education_table()
    )

    data_inse.create_basic_education_table()
    
    X_train_scaled, X_test_scaled, y_train, y_test = data_inse.prepare_data(inse_with_inep)

    # Treinar modelo
    model = MLPModel(input_dim=X_train_scaled.shape[1])
    history = model.train(X_train_scaled, y_train)
    
    # Fazer predições e avaliar
    predictions = model.predict(X_test_scaled)
    loss, mae = model.evaluate(X_test_scaled, y_test)
    
    # Visualizar resultados
    visualizer = ModelVisualizer()
    visualizer.plot_learning_curve(history)
    r2 = visualizer.plot_predictions_vs_real(y_test, predictions)
    visualizer.plot_error_distribution(y_test, predictions)
    visualizer.plot_confusion_matrix(y_test, predictions, threshold=y_test.mean())
    
    metrics = {
        'MAE': mae,
        'R²': r2,
        'MSE': loss
    }

    print("Métricas do Modelo MLP:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    visualizer.plot_metrics(metrics)

    # Treinar modelo Random Forest
    print("Métricas do Random Forest:")

    rf_model = RandomForestModel()
    rf_model.train(X_train_scaled, y_train)
    rf_predictions = rf_model.predict(X_test_scaled)
    rf_metrics = rf_model.evaluate(X_test_scaled, y_test)
    rf_visualizer = ModelVisualizer()
    rf_r2 = rf_visualizer.plot_predictions_vs_real_rf(y_test, rf_predictions)
    rf_visualizer.plot_error_distribution_rf(y_test, rf_predictions)
    rf_visualizer.plot_confusion_matrix_rf(y_test, rf_predictions, threshold=y_test.mean())
    rf_metrics_dict = {
        'MAE': rf_metrics['mae'],
        'R²': rf_metrics['r2'],
        'MSE': rf_metrics['mse']
    }
    rf_visualizer.plot_metrics_rf(rf_metrics_dict)

if __name__ == "__main__":
    main()