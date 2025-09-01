import os
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib

# Intentamos usar TkAgg (interactivo); si falla (entorno sin GUI) caemos a Agg.
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

# ---- Funciones propias del proyecto (igual que en tu base) ----
from scripts.preprocess_data_Tetuan import preprocess_data_electricity
from scripts.buscarIndexTetuan import fila_para_fecha


# ===================== CONFIG =====================
MODELS_DIR = "models"
DATA_CSV   = "data/02_Tetuan_City_power_consumption.csv"

# Coincide con el entrenamiento de tus modelos
SEQ_LENGTH = 48

# Ventana de predicción (a 10 min -> 6 por hora)
DIAS_A_PREDECIR = 7
NUM_PRED = int(6 * 24 * DIAS_A_PREDECIR)

# Punto de arranque: por fecha o por índice
DIA, MES, ANIO = 6, 3, 2017
indice_inicial = fila_para_fecha(DIA, MES, ANIO)
if 0:
    indice_inicial += 6*12  # offset para probar otro punto desde el medio dia

# Alternativa: fija a mano
# indice_inicial = 17858

# ==================================================


def find_model_files(models_dir: str):
    """
    Busca ficheros de modelo que TERMINEN en '_1' seguido de extensión válida.
    DL: .h5 / .keras
    ML: .pkl
    """
    files = []
    for fname in os.listdir(models_dir):
        # Deben terminar en _1.(h5|keras|pkl)
        if re.search(r"_1\.(h5|keras|pkl)$", fname, re.IGNORECASE):
            full = os.path.join(models_dir, fname)
            if os.path.isfile(full):
                files.append(full)
    # Orden para que siempre sea reproducible
    return sorted(files)


def is_deep_learning_model(path: str) -> bool:
    return path.lower().endswith((".h5", ".keras"))


def load_any_model(path: str):
    if is_deep_learning_model(path):
        # Cuidamos custom_objects por si usaste 'mse' en compile
        model = load_model(path, compile=False, custom_objects={"mse": tf.keras.losses.mse})
        model_type = "DL"
    else:
        model = joblib.load(path)
        model_type = "ML"
    return model, model_type


def inverse_target(scaler_y, y_pred_scaled):
    """
    Invierte el escalado de y suponiendo que el scaler_y espera 3 columnas
    (como en tu base): target + 2 dummy.
    y_pred_scaled: array shape (1,) o (1,1) o (n,1)
    Devuelve float(s) en escala original.
    """
    y = np.atleast_2d(np.array(y_pred_scaled).reshape(-1, 1))  # (n,1)
    y3 = np.hstack([y, np.zeros((y.shape[0], 2))])             # (n,3)
    inv = scaler_y.inverse_transform(y3)[:, 0]
    return inv


def predict_series_for_model(model, model_type, X_exog, y_scaled, scaler_y,
                             start_index, seq_len, num_steps):
    """
    Calcula valores reales + predicción directa + autoregresiva para un modelo.
    Devuelve: valores_reales, pred_directa, pred_auto
    """
    n = X_exog.shape[0]
    valores_reales = []
    pred_directa = []
    pred_auto = []

    # Guardamos las predicciones en ESCALA y para autoregresivo (para realimentar)
    pred_auto_scaled_buffer = []

    for k in range(num_steps):
        start_idx = start_index + k
        end_idx = start_idx + seq_len + 1
        if end_idx >= n:
            break

        # ====== Construcción secuencia DIRECTA (usa sólo reales) ======
        seq_dir = np.concatenate(
            [X_exog[start_idx:end_idx, 0:1], X_exog[start_idx:end_idx, 3:]],
            axis=1
        ).copy()
        seq_dir[-1, 0] = 0  # incógnita

        if model_type == "DL":
            X_in = seq_dir.reshape(1, seq_len + 1, seq_dir.shape[1])
        else:
            X_in = seq_dir.reshape(1, -1)

        y_pred_scaled_direct = model.predict(X_in)
        # Normalizamos formatos (a (1,1) -> (1,))
        y_pred_scaled_direct = np.array(y_pred_scaled_direct).reshape(-1, 1)

        # Inversa para pintar
        y_pred_direct = inverse_target(scaler_y, y_pred_scaled_direct)[0]

        # Valor real (desescalado)
        y_real = scaler_y.inverse_transform(
            y_scaled[end_idx - 1].reshape(1, -1)
        )[0, 0]

        valores_reales.append(float(y_real))
        pred_directa.append(float(y_pred_direct))

        # ====== Construcción secuencia AUTORREGRESIVA ======
        seq_auto = np.concatenate(
            [X_exog[start_idx:end_idx, 0:1], X_exog[start_idx:end_idx, 3:]],
            axis=1
        ).copy()

        # Sustituimos hasta SEQ_LENGTH valores previos por predicciones propias (ESCALA y)
        max_back = min(k, seq_len)
        for j in range(1, max_back + 1):
            seq_auto[-(j + 1), 0] = float(np.array(pred_auto_scaled_buffer[-j]).reshape(()))

        seq_auto[-1, 0] = 0  # incógnita

        if model_type == "DL":
            X_in_auto = seq_auto.reshape(1, seq_len + 1, seq_auto.shape[1])
        else:
            X_in_auto = seq_auto.reshape(1, -1)

        y_pred_scaled_auto = model.predict(X_in_auto)
        y_pred_scaled_auto = np.array(y_pred_scaled_auto).reshape(-1, 1)  # (1,1) -> (1,1)
        pred_auto_scaled_buffer.append(float(y_pred_scaled_auto[0, 0]))

        # Para pintar, desescalamos
        y_pred_auto_real = inverse_target(scaler_y, y_pred_scaled_auto)[0]
        pred_auto.append(float(y_pred_auto_real))

    return np.array(valores_reales), np.array(pred_directa), np.array(pred_auto)


def main():
    # -------- Carga datos y escaladores --------
    print("Cargando dataset y preprocesando...")
    df = pd.read_csv(DATA_CSV)
    X_exog, y_scaled, _, _ = preprocess_data_electricity(df)

    print("Cargando scalers...")
    scaler_X = joblib.load(os.path.join(MODELS_DIR, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(MODELS_DIR, "scaler_y.pkl"))

    # -------- Descubre modelos --------
    model_files = find_model_files(MODELS_DIR)
    if not model_files:
        raise RuntimeError(f"No se encontraron modelos que terminen en '_1' en {MODELS_DIR}")

    print("Modelos detectados:")
    for f in model_files:
        print("  -", os.path.basename(f))

    # -------- Ejecuta predicciones para cada modelo --------
    resultados = {}  # nombre_modelo -> dict con arrays
    valores_reales_ref = None
    min_len = None

    for path in model_files:
        model, mtype = load_any_model(path)
        name = os.path.splitext(os.path.basename(path))[0]

        print(f"\nProcesando {name} ({mtype}) ...")
        reales, pdir, paut = predict_series_for_model(
            model, mtype, X_exog, y_scaled, scaler_y,
            start_index=indice_inicial,
            seq_len=SEQ_LENGTH,
            num_steps=NUM_PRED
        )

        # Guardamos y ajustamos longitud común
        resultados[name] = {
            "tipo": mtype,
            "directa": pdir,
            "autoregresiva": paut
        }

        if valores_reales_ref is None:
            valores_reales_ref = reales
            min_len = len(reales)
        else:
            min_len = min(min_len, len(reales), len(valores_reales_ref))

        min_len = min(min_len, len(pdir), len(paut))

    # Recorta todo al mínimo común para superponer correctamente
    pasos = np.arange(min_len)
    valores_reales_ref = valores_reales_ref[:min_len]

    for k in resultados:
        resultados[k]["directa"] = resultados[k]["directa"][:min_len]
        resultados[k]["autoregresiva"] = resultados[k]["autoregresiva"][:min_len]

    # -------- Métricas rápidas (MAE, MSE, RMSE) --------
    print("\nMAE (Directa / Autoregresiva):")
    for k, v in resultados.items():
        mae_dir = np.mean(np.abs(valores_reales_ref - v["directa"]))
        mae_aut = np.mean(np.abs(valores_reales_ref - v["autoregresiva"]))
        print(f"  {k:30s}  {mae_dir:8.3f} / {mae_aut:8.3f}")


    print("\nMSE (Directa / Autoregresiva):")
    for k, v in resultados.items():
        mse_dir = np.mean((valores_reales_ref - v["directa"]) ** 2)
        mse_aut = np.mean((valores_reales_ref - v["autoregresiva"]) ** 2)
        print(f"  {k:30s}  {mse_dir:8.3f} / {mse_aut:8.3f}")
   
    print("\nRMSE (Directa / Autoregresiva):")
    for k, v in resultados.items():
        rmse_dir = np.sqrt(np.mean((valores_reales_ref - v["directa"]) ** 2))
        rmse_aut = np.sqrt(np.mean((valores_reales_ref - v["autoregresiva"]) ** 2))
        print(f"  {k:30s}  {rmse_dir:8.3f} / {rmse_aut:8.3f}")

    # -------- Gráficas --------
    plt.figure(figsize=(12, 7))
    plt.plot(pasos, valores_reales_ref, label="Real", linewidth=2)

    for k, v in resultados.items():
        plt.plot(pasos, v["directa"], linestyle="--", label=f"{k} (Directa)")
    
    plt.title(f"Direct prediction from {DIA}/{MES}/{ANIO} - Prediction days: {DIAS_A_PREDECIR:.0f}")
    plt.xlabel("Step")
    plt.ylabel("kWh")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.plot(pasos, valores_reales_ref, label="Real", linewidth=2)

    for k, v in resultados.items():
        plt.plot(pasos, v["autoregresiva"], linestyle="--", label=f"{k} (Autoregresiva)")
    
    plt.title(f"Autoregressive prediction from {DIA}/{MES}/{ANIO} - Prediction days: {DIAS_A_PREDECIR:.0f}")
    plt.xlabel("Step")
    plt.ylabel("kWh")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    aa=1

if __name__ == "__main__":
    main()
