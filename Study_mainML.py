import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error
from time import time # Importa time para medir el tiempo de entrenamiento

# Importar la función de preprocesamiento
from scripts.generacionModelosLSTM import seleccionModelos # Importar la función que genera los modelos LSTM para el estudio de ablación
from scripts.preprocess_data_Tetuan import preprocess_data_electricity

#Modelos ML
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Cargar datos
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')

# Preprocesar datos
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)


def splitData(data):
    sequences = []
    labels = []
    sequences = data[:, 1:]
    # Suponemos que el valor a predecir
    labels = data[:, 0]
    return sequences, labels

X_seq, y_seq = splitData(X_exog)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)


#-
# Definir los hiperparámetros para cada modelo
linear_params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
gb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 5, 7], 'subsample': [0.7, 0.8, 1.0]}
xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 5, 7], 'subsample': [0.7, 0.8, 1.0], 'gamma': [0, 0.1, 0.5, 1]}
svr_params = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf'], 'epsilon': [0.01, 0.1, 0.2]}

# Definir un diccionario con los modelos
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(),
    'SVR': SVR()
}

# Definir un diccionario con los hiperparámetros
params = {
    'Linear Regression': linear_params,
    'Ridge Regression': linear_params,
    'Lasso Regression': linear_params,
    'Random Forest': rf_params,
    'Gradient Boosting': gb_params,
    'XGBoost': xgb_params,
    'SVR': svr_params
}


# Lista para almacenar los resultados
results = []
# Bucle para probar todas las combinaciones de hiperparámetros para cada modelo
for model_name, model in models.items():
    if model_name == 'Linear Regression':
        grid = [{}]  # no se modifican los parámetros
    else:
        grid = list(ParameterGrid(params[model_name]))
    for param_set in grid:
        model.set_params(**param_set)

        # Medir el tiempo de entrenamiento
        start_time = time.time()
        
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Medir el tiempo de fin
        end_time = time.time()
        training_time = end_time - start_time
        
        # Hacer predicciones
        predictions = model.predict(X_test)

        # Suponiendo que scaler_y fue ajustado en la variable target original
        y_test_real = scaler_y.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 2)))))[:, 0]
        predictions_real = scaler_y.inverse_transform(np.hstack((predictions.reshape(-1, 1), np.zeros((len(predictions), 2)))))[:, 0]

        # Evaluación del modelo
        mse = mean_squared_error(y_test_real, predictions_real)
        r2 = r2_score(y_test_real, predictions_real)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
        mae = np.mean(np.abs(y_test_real - predictions_real))
        mre = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
        cvrmse = np.std((y_test_real - predictions_real) / y_test_real) * 100
        # Guardar los resultados
        results.append({
            'Model': model_name,
            'Params': param_set,
            'MSE': mse,
            'R2': r2,
            'MAE': mae,
            'MAPE': mape,
            'MRE': mre,
            #'loss': loss,
            'RMSE': rmse,
            'CVRMSE': cvrmse,
            'Training Time (s)': training_time
        })

        # Convertir los resultados a un DataFrame
        results_df = pd.DataFrame(results)

        # Guardar los resultados en un archivo CSV
        results_df.to_csv('./output/ResultsML/ML_ablation_results_Zone1.csv', index=False)

        print(model_name, param_set)

print("Resultados guardados en 'ML_ablation_results.csv'")
