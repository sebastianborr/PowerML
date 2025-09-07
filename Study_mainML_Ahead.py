import os
import io
import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib

# --- Import de tu pipeline ---
from scripts.preprocess_data_Tetuan import preprocess_data_electricity

# --------------- CONFIGURACIÓN HIPERPARÁMETROS ----------------
# Ventana base e HORIZONTES (puedes cambiarlos)
SEQ_LENGTH = 48                              # Ventana base de entrada
PREDICTION_HORIZONS = [1, 6, 12, 24, 48]     # +10min, +1h, +2h, +4h, +8h (si tu frecuencia es 10min)

# Identificador de estudio (solo para nombres de salida)
NUM_CONFIGURACION_ESTUDIO = 115

# ====== ELIGE EL MODELO (UNO SOLO) ======
# Opciones: 'GRADIENT_BOOSTING' o 'XGBOOST'
MODEL_FAMILY = 'XGBOOST'   # <- cámbialo a 'GRADIENT_BOOSTING' si quieres GBR

# ---- CONFIGURACIÓN DEL MODELO (edita aquí tus mejores hiperparámetros) ----
# Gradient Boosting (scikit-learn)
GB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    subsample=0.8,
)

# XGBoost (xgboost.XGBRegressor)
XGB_USE_GPU = True  # si no tienes GPU, pon False
XGB_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    subsample=1,
    colsample_bytree=0.8,
    gamma=0,
    gpu_id=0 if XGB_USE_GPU else None
)

# ------------------ CARGA Y PREPROCESADO ------------------
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)

# ------------------ HELPERS ------------------
def inv_y(scaler_y, arr_1d):
    """Invierte la normalización del target (univar) para métricas legibles."""
    arr_1d = np.asarray(arr_1d).reshape(-1, 1)
    return scaler_y.inverse_transform(
        np.hstack((arr_1d, np.zeros((len(arr_1d), 2))))
    )[:, 0]

def create_direct_sequences_with_mask(data, seq_length, horizon, target_index=0):
    """
    Construye muestras directas (un modelo por h):
      - Entrada: ventana extendida (seq_length + h) con y futura enmascarada.
      - Salida: y en t + h (escalar).
    Usa y en col 0 y exógenas a partir de col 3 (como en tus scripts).
    """
    X_list, y_list = [], []
    n = len(data)
    for i in range(n - seq_length - horizon + 1):
        window = np.concatenate(
            [data[i:i+seq_length+horizon, 0:1],     # y
             data[i:i+seq_length+horizon, 3:]],     # exógenas
            axis=1
        ).copy()
        window[-horizon:, 0] = 0  # enmascarar y futura

        X_list.append(window.flatten())
        y_list.append(data[i + seq_length + horizon - 1, target_index])

    return np.array(X_list), np.array(y_list)

def build_estimator():
    """Crea el estimador según MODEL_FAMILY y params fijos arriba."""
    if MODEL_FAMILY.upper() == 'GRADIENT_BOOSTING':
        return GradientBoostingRegressor(**GB_PARAMS)
    elif MODEL_FAMILY.upper() == 'XGBOOST':
        # filtra None para gpu_id si no se usa GPU
        params = {k: v for k, v in XGB_PARAMS.items() if v is not None}
        return XGBRegressor(**params)
    else:
        raise ValueError("MODEL_FAMILY debe ser 'GRADIENT_BOOSTING' o 'XGBOOST'.")

def try_save_h5(model, path_h5):
    """
    Guarda el modelo dentro de un contenedor HDF5 (.h5) como bytes serializados (joblib).
    Requiere 'h5py'. Si no está o falla, no interrumpe el flujo.
    """
    try:
        import h5py
        buf = io.BytesIO()
        joblib.dump(model, buf)
        data = np.frombuffer(buf.getvalue(), dtype='uint8')
        with h5py.File(path_h5, 'w') as f:
            f.create_dataset('joblib_pickle', data=data)
            f.attrs['framework'] = 'sklearn/xgboost'
            f.attrs['model_family'] = MODEL_FAMILY
        return True
    except Exception as e:
        print(f"[Aviso] No se pudo guardar .h5: {e}")
        return False

# ------------------ SALIDAS ------------------
OUT_DIR_MODELS = "models"
OUT_DIR_RESULTS = "./output/AheadML"
os.makedirs(OUT_DIR_MODELS, exist_ok=True)
os.makedirs(OUT_DIR_RESULTS, exist_ok=True)

# =======================
#      BUCLE DE ESTUDIO
# =======================
results = []
for pred_len in PREDICTION_HORIZONS:
    print(f"Entrenando {MODEL_FAMILY} para predicción directa a h={pred_len}...")

    # Construcción de secuencias directas (SEQ_LENGTH + h)
    X_seq, y_seq = create_direct_sequences_with_mask(X_exog, SEQ_LENGTH, pred_len)

    # Split (mismo estilo que tu script Keras): 80/20 aleatorio
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    # Modelo (un único set de hiperparámetros ya conocidos)
    model = build_estimator()

    # Entrenamiento
    t0 = time()
    # Si quisieras early stopping en XGB:
    # if MODEL_FAMILY.upper() == 'XGBOOST':
    #     model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
    #               eval_metric='rmse', verbose=False, early_stopping_rounds=50)
    # else:
    model.fit(X_train, y_train)
    training_time = time() - t0

    # Predicciones
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)

    # Desnormalización para métricas (como en tus scripts)
    y_train_real = inv_y(scaler_y, y_train)
    y_valid_real = inv_y(scaler_y, y_valid)
    pred_train_real = inv_y(scaler_y, pred_train)
    pred_valid_real = inv_y(scaler_y, pred_valid)

    # Métricas agregadas en VALIDACIÓN
    valid_mse  = mean_squared_error(y_valid_real, pred_valid_real)
    valid_rmse = np.sqrt(valid_mse)
    r2         = r2_score(y_valid_real, pred_valid_real)
    mae        = np.mean(np.abs(y_valid_real - pred_valid_real))
    # evitar /0
    eps = 1e-8
    mape       = np.mean(np.abs((y_valid_real - pred_valid_real) / (y_valid_real + eps))) * 100
    cvrmse     = np.std((y_valid_real - pred_valid_real) / (y_valid_real + eps)) * 100

    # "history-like" losses (RMSE en train y valid)
    train_mse  = mean_squared_error(y_train_real, pred_train_real)
    train_rmse = np.sqrt(train_mse)
    train_loss_final = train_rmse
    val_loss_final   = valid_rmse
    overfit_gap      = val_loss_final - train_loss_final
    loss_ratio       = (val_loss_final / train_loss_final) if train_loss_final != 0 else np.nan

    # Registro de resultados
    result_row = {
        'modelFamily': MODEL_FAMILY,
        'prediction_horizon': pred_len,
        'MSE': valid_mse,
        'RMSE': valid_rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'CVRMSE': cvrmse,
        'Training Time (s)': training_time,
        'Train Loss Final': train_loss_final,
        'Validation Loss Final': val_loss_final,
        'Overfit Gap': overfit_gap,
        'Loss Ratio': loss_ratio
    }

    # añade hiperparámetros usados (para traza)
    if MODEL_FAMILY.upper() == 'GRADIENT_BOOSTING':
        result_row.update({f"GradientBoost_{k}": v for k, v in GB_PARAMS.items()})
    else:
        result_row.update({f"XGBoost_{k}": v for k, v in XGB_PARAMS.items() if v is not None})

    results.append(result_row)

    # Guardar modelo por horizonte
    base_name = f"{MODEL_FAMILY}_{NUM_CONFIGURACION_ESTUDIO}_{pred_len}"
    path_joblib = os.path.join(OUT_DIR_MODELS, base_name + ".joblib")
    joblib.dump(model, path_joblib)

    # Copia .h5 opcional
    # path_h5 = os.path.join(OUT_DIR_MODELS, base_name + ".h5")
    # try_save_h5(model, path_h5)

    # Para XGBoost, guardado nativo adicional (.json), útil para despliegue
    if MODEL_FAMILY.upper() == 'XGBOOST':
        try:
            model.get_booster().save_model(os.path.join(OUT_DIR_MODELS, base_name + ".json"))
        except Exception as e:
            print(f"[Aviso] No se pudo guardar JSON nativo de XGBoost: {e}")

    # Guardado incremental de resultados
    out_csv = os.path.join(OUT_DIR_RESULTS, f"{MODEL_FAMILY}_multistep_prediction_results_{NUM_CONFIGURACION_ESTUDIO}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)

print(f"Estudio {MODEL_FAMILY}-ahead completado.")
