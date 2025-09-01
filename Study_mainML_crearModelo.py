import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import time
import joblib

# Cargar datos ya preprocesados
from scripts.preprocess_data_Tetuan import preprocess_data_electricity

# ===============================
# Cargar y preparar los datos
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)
X_clean = np.concatenate([X_exog[:, 0:1], X_exog[:, 3:]], axis=1)
# ===============================
# Función para generar secuencias con y enmascarado
def create_ML_sequences_with_masked_target(data, target_index, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data) - 1):
        window = data[i - seq_length:i + 1].copy()  # ventana + paso actual
        window[-1, target_index] = 0  # enmascarar y(t)
        X.append(window.flatten())
        y.append(data[i, target_index])  # target real y(t)
    return np.array(X), np.array(y)

# ===============================

# Helper: invertir normalización del target
def inv_y(scaler_y, arr):
    arr = np.asarray(arr).reshape(-1, 1)
    return scaler_y.inverse_transform(
        np.hstack((arr, np.zeros((len(arr), 2))))
    )[:, 0]


# Crear secuencias y dividir temporalmente
seq_len = 48
X_seq, y_seq = create_ML_sequences_with_masked_target(X_clean, target_index=0, seq_length=seq_len)

split_index = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]
# ===============================
# Modelos y sus grids
#models = {
#    'Linear Regression': LinearRegression(),
#    'Ridge Regression': Ridge(),
#    'Lasso Regression': Lasso(),
#    'Random Forest': RandomForestRegressor(),
#    'Gradient Boosting': GradientBoostingRegressor(),
#    'XGBoost': XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0),
#    'SVR': SVR()
#}
#
#params = {
#    'Linear Regression': [{}],
#    'Ridge Regression': {'alpha': [0.01, 0.1, 1, 10]},
#    'Lasso Regression': {'alpha': [0.01, 0.1, 1]},
#    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]},
#    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 5, 7], 'subsample': [0.7, 0.8, 1.0]},
#    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 5, 7], 'subsample': [0.7, 0.8, 1.0], 'gamma': [0, 0.1, 0.5, 1]},
#    'SVR': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf'], 'epsilon': [0.01, 0.1, 0.2]},
#}
models = {
    'Linear Regression': LinearRegression()
}

params = {
    'Linear Regression': [{}]
}


# ===============================
# Entrenamiento y evaluación
results = []

for model_name, model in models.items():
    grid = list(ParameterGrid(params[model_name]))

    for param_set in grid:
        model.set_params(**param_set)

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

         # --- predicciones en train (para Train_Loss_Final) y valid (para todo lo demás) ---
        preds_train = model.predict(X_train)
        preds_valid = model.predict(X_test)

        # --- desescalar ---
        y_train_real     = inv_y(scaler_y, y_train)
        y_valid_real     = inv_y(scaler_y, y_test)
        preds_train_real = inv_y(scaler_y, preds_train)
        preds_valid_real = inv_y(scaler_y, preds_valid)

        # === pérdidas estilo "history": usamos RMSE como loss ===
        train_mse  = mean_squared_error(y_train_real, preds_train_real)
        valid_mse  = mean_squared_error(y_valid_real, preds_valid_real)
        train_rmse = np.sqrt(train_mse)
        valid_rmse = np.sqrt(valid_mse)

        Train_Loss_Final = train_rmse
        Val_Loss_Final   = valid_rmse
        Overfit_Gap      = Val_Loss_Final - Train_Loss_Final
        Loss_Ratio       = (Val_Loss_Final / Train_Loss_Final) if Train_Loss_Final != 0 else np.nan

        # === métricas sobre VALIDACIÓN (no sobre test) ===
        r2      = r2_score(y_valid_real, preds_valid_real)
        mae     = np.mean(np.abs(y_valid_real - preds_valid_real))
        mape    = np.mean(np.abs((y_valid_real - preds_valid_real) / y_valid_real)) * 100
        cvrmse  = np.std((y_valid_real - preds_valid_real) / y_valid_real) * 100

        results.append({
            'Model': model_name,
            'Params': param_set,
            # métricas validación
            'MSE': valid_mse,
            'RMSE': valid_rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'CVRMSE': cvrmse,
            # extras
            'Training Time (s)': training_time,
            # bloque tipo history
            'Train_Loss_Final': Train_Loss_Final,
            'Val_Loss_Final': Val_Loss_Final,
            'Overfit_Gap': Overfit_Gap,
            'Loss_Ratio': Loss_Ratio

        })

        print(f"{model_name} - {param_set}")
        joblib.dump(model, 'models/ML_RL.pkl')
        # Guardar resultados
        #results_df = pd.DataFrame(results)
        #results_df.to_csv('./output/ResultsML/01_ML_ablation_results_perdidas.csv', index=False)


print("Resultados guardados.") 

# ===============================