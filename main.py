import os
import argparse
import pandas as pd
from src.data.loader import DataLoader
from src.models.mlp_regressor import MLPRegressorModel
from src.models.gamma_regressor import GammaRegressorModel
from src.visualization.plots_regressao import RegressorVisualizer
from src.models.random_forest_regressor import RandomForestRegressorModel
from src.models.mlp_classifier_model import MLPModel as MLPClassifierModel
from src.visualization.plots import ModelVisualizer as ClassifierVisualizer
from src.models.random_forest_model import RandomForestModel as RandomForestClassifierModel


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
    
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de análise de evasão escolar."
    )

    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['class', 'reg'],
        help="Define o modo de operação: 'class' (quartis) ou 'reg' (valor contínuo)."
    )
    args = parser.parse_args()
    MODE = args.mode
    print(f"--- EXECUTANDO EM MODO: {MODE.upper()} ---")
    
    # Inicializa o carregador de dados com o arquivo de transição
    data_inse = DataLoader('data/TX_TRANSICAO_MUNICIPIOS_2021_2022.xlsx')
    
    # Lógica de cache
    if os.path.exists('data/data_combined.csv'):
        print("Carregando dados combinados do arquivo CSV existente...")
        inse_with_inep = pd.read_csv('data/data_combined.csv', encoding='utf-8-sig')
    else:
        # Carrega todas as tabelas
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
    
    # Prepara os dados de acordo com o modo
    X_train_scaled, X_test_scaled, y_train, y_test = data_inse.prepare_data(
        inse_with_inep, mode=MODE
    )

    ###########################################
    #       2. MODO DE CLASSIFICAÇÃO          #
    ###########################################
    
    if MODE == 'class':
        print("\n=== Análise do Modelo MLP (Classificação) ===")
        # Converte labels para inteiros para o MLP
        y_train_int = y_train.astype('category').cat.codes
        y_test_int = y_test.astype('category').cat.codes
        
        # --- CORREÇÃO DE TYPO: MLPClassifierModel ---
        model = MLPClassifierModel(input_dim=X_train_scaled.shape[1])
        history = model.train(X_train_scaled, y_train_int)
        mlp_metrics = model.evaluate(X_test_scaled, y_test_int)
        mlp_predictions = model.predict(X_test_scaled)
        
        metrics = {
            'Acurácia': mlp_metrics['accuracy'],
            'F1-Score (Macro)': mlp_metrics['f1_score_macro'],
            'Loss': mlp_metrics['loss']
        }
        
        print("\nMétricas do Modelo MLP:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # --- CORREÇÃO DE INSTÂNCIA: ClassifierVisualizer ---
        visualizer = ClassifierVisualizer()
        visualizer.plot_learning_curve(history)
        visualizer.plot_confusion_matrix(y_test_int, mlp_predictions)
        
        print("Gerando Curva ROC para o MLP...")
        mlp_probabilities = model.predict_proba(X_test_scaled)
        visualizer.plot_roc_curve(
            y_test_int,             # Os labels verdadeiros (0, 1, 2)
            mlp_probabilities,      # As probabilidades
            model.class_labels,     # A lista [0, 1, 2]
            model_name='MLP'
        )

        print("\n=== Análise do Random Forest (Classificação) ===")
        rf_model = RandomForestClassifierModel()
        # Mude para tune=True se quiser rodar a otimização
        rf_model.train(X_train_scaled, y_train, tune=False) 
        rf_predictions = rf_model.predict(X_test_scaled)
        rf_metrics = rf_model.evaluate(X_test_scaled, y_test)
        rf_metrics_dict = {
            'Acurácia': rf_metrics['accuracy'],
            'F1-Score (Macro)': rf_metrics['f1_score_macro']
        }
        
        # --- CORREÇÃO DE INSTÂNCIA: ClassifierVisualizer ---
        rf_visualizer = ClassifierVisualizer()
        
        # Assumindo que você tem uma função 'plot_metrics' no seu ClassifierVisualizer
        # Se ela se chamar 'plot_metrics_rf', mude aqui.
        rf_visualizer.plot_metrics(rf_metrics_dict) 
        
        rf_visualizer.plot_confusion_matrix_rf(y_test, rf_predictions)
        
        print("Gerando Curva ROC para o Random Forest...")
        rf_probabilities = rf_model.predict_proba(X_test_scaled)
        rf_visualizer.plot_roc_curve(
            y_test,                 # Os labels verdadeiros (texto)
            rf_probabilities,       # As probabilidades
            rf_model.class_labels,  # A lista de textos (ex: 'Alta Evasão'...)
            model_name='Random Forest'
        )
        
        print("\n--- Gerando Gráfico de Features (Classificação) ---")
        feature_names = data_inse.feature_names
        importances_df = rf_model.get_feature_importances(feature_names)
        print(importances_df.head(10).to_string())
        # Chama a nova função de plotagem
        rf_visualizer.plot_feature_importance(
            importances_df,
            model_name='Random_Forest_Classifier'
        )
        

    ###########################################
    #         3. MODO DE REGRESSÃO            #
    ###########################################
    
    elif MODE == 'reg':
        
        # --- CORREÇÃO DE INSTÂNCIA: RegressorVisualizer ---
        reg_viz = RegressorVisualizer()
        # O visualizer do classificador (para a curva de aprendizado do MLP)
        class_viz = ClassifierVisualizer()
        
        print("\n=== Análise do Modelo MLP (Regressão) ===")
        mlp_reg = MLPRegressorModel(input_dim=X_train_scaled.shape[1])
        mlp_history = mlp_reg.train(X_train_scaled, y_train)
        mlp_reg_metrics = mlp_reg.evaluate(X_test_scaled, y_test)

        # --- Plots do MLP ---
        mlp_predictions = mlp_reg.predict(X_test_scaled)
        class_viz.plot_learning_curve(mlp_history)
        reg_viz.plot_predictions_vs_real(y_test, mlp_predictions, model_name='MLP')
        reg_viz.plot_error_distribution(y_test, mlp_predictions, model_name='MLP')
        reg_viz.plot_metrics(mlp_reg_metrics, model_name='MLP')
        

        print("\n=== Análise do Random Forest (Regressão) ===")
        rf_reg = RandomForestRegressorModel()
        rf_reg.train(X_train_scaled, y_train, tune=False) 
        rf_reg_metrics = rf_reg.evaluate(X_test_scaled, y_test)
        
        # --- Plots do RF ---
        rf_predictions = rf_reg.predict(X_test_scaled)
        reg_viz.plot_predictions_vs_real(y_test, rf_predictions, model_name='Random_Forest')
        reg_viz.plot_error_distribution(y_test, rf_predictions, model_name='Random_Forest')
        reg_viz.plot_metrics(rf_reg_metrics, model_name='Random_Forest')
        
        print("\n--- Gerando Gráfico de Features (Regressão) ---")
        feature_names = data_inse.feature_names
        importances_df = rf_reg.get_feature_importances(feature_names)
        print(importances_df.head(10).to_string())
        # Chama a nova função de plotagem
        reg_viz.plot_feature_importance(
            importances_df, 
            model_name='Random_Forest_Regressor'
        )
        
        # print("\n=== Análise do Gamma Regressor ===")
        # gamma_reg = GammaRegressorModel()
        # gamma_reg.train(X_train_scaled, y_train, tune=False)
        # gamma_reg_metrics = gamma_reg.evaluate(X_test_scaled, y_test)
        
        # # --- Plots do Gamma ---
        # gamma_predictions = gamma_reg.predict(X_test_scaled)
        # reg_viz.plot_predictions_vs_real(y_test, gamma_predictions, model_name='Gamma')
        # reg_viz.plot_error_distribution(y_test, gamma_predictions, model_name='Gamma')
        # reg_viz.plot_metrics(gamma_reg_metrics, model_name='Gamma')
        
        # print("\n--- 10 Features Mais Importantes (Gamma Regressor) ---")
        # importances_df_gamma = gamma_reg.get_feature_importances(feature_names)
        # print(importances_df_gamma.head(10).to_string())

if __name__ == "__main__":
    main()