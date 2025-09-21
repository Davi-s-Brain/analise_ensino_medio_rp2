from src.data.loader import DataLoader
from src.models.mlp_model import MLPModel
from src.visualization.plots import ModelVisualizer

def main():
    # Carregar e preparar dados
    data_loader = DataLoader('TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx')
    df = data_loader.load_data()
    X_train_scaled, X_test_scaled, y_train, y_test = data_loader.prepare_data(df)

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
    visualizer.plot_metrics(metrics)

if __name__ == "__main__":
    main()