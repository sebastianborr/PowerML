import os
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib
import types  # para enlazar el método correctamente
# Intentamos usar TkAgg (interactivo); si falla (entorno sin GUI) caemos a Agg.
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---- Funciones propias del proyecto ----
from scripts.preprocess_data_Tetuan import preprocess_data_electricity
from scripts.buscarIndexTetuan import fila_para_fecha
matplotlib.rcParams['savefig.directory'] = r'/mnt/c/Users/Sebas/Desktop/PowerML/output/ImagesVarias/cap4_ahead'  # cambia si quieres

# ===================== CONFIG =====================
NumPrueba = 6                 # identificador en pantalla
NumAhead  = 48                 # <<<< AHEAD SELECCIONADO (p.e. 1, 6, 12, 24, 48)
ploteoGraficas = True

DIA, MES, ANIO = 20,12, 2017  # punto de arranque por fecha
indice_inicial = fila_para_fecha(DIA, MES, ANIO)

MODELS_DIR = "models"
DATA_CSV   = "data/02_Tetuan_City_power_consumption.csv"

SEQ_LENGTH = 48               # igual que en entrenamiento
DIAS_A_PREDECIR = 7
NUM_PUNTOS = int(6 * 24 * DIAS_A_PREDECIR)  # puntos de la serie que quieres evaluar/pintar
# ==================================================

def find_model_files(models_dir: str, ahead: int):
    files = []
    patron = re.compile(rf"_{ahead}\.(h5|keras|pkl|joblib)$", re.IGNORECASE)
    for fname in os.listdir(models_dir):
        if patron.search(fname):
            full = os.path.join(models_dir, fname)
            if os.path.isfile(full):
                files.append(full)
    return sorted(files)

def is_deep_learning_model(path: str) -> bool:
    return path.lower().endswith((".h5", ".keras"))

def load_any_model(path: str):
    if is_deep_learning_model(path):
        model = load_model(path, compile=False, custom_objects={"mse": tf.keras.losses.mse})
        mtype = "DL"
    else:
        model = joblib.load(path)
        mtype = "ML"
    return model, mtype

def inverse_block(scaler_y, block_scaled_vec):
    block_scaled_vec = np.array(block_scaled_vec).reshape(-1)         # (NumAhead,)
    mat = np.zeros((len(block_scaled_vec), 3))                        # (NumAhead, 3)
    mat[:, 0] = block_scaled_vec
    inv = scaler_y.inverse_transform(mat)[:, 0]                       # (NumAhead,)
    return inv

# ---------- NUEVO: hace una predicción de bloque en ESCALA ----------
def predict_block_scaled(model, mtype, seq_mat, seq_len, ahead):
    """
    Devuelve SIEMPRE un vector (ahead,) en ESCALA.
    - Si el modelo devuelve (ahead,), lo usamos tal cual.
    - Si devuelve (1,), iteramos 'ahead' veces dentro del bloque,
      realimentando cada predicción en la ventana (posición -ahead + h).
    - Si devolviera >ahead, cogemos los últimos 'ahead'; si <ahead (y !=1), rellenamos.
    """
    def _call(m, mt, s):
        X = s.reshape(1, seq_len + ahead, s.shape[1]) if mt == "DL" else s.reshape(1, -1)
        y = np.array(m.predict(X)).reshape(-1)
        return y

    y0 = _call(model, mtype, seq_mat)
    if y0.size == ahead:
        return y0

    if y0.size == 1:
        preds = []
        seq_iter = seq_mat.copy()
        for h in range(ahead):
            yh = _call(model, mtype, seq_iter).reshape(-1)
            yh = float(yh[0])
            preds.append(yh)
            # realimenta en la posición correspondiente del tramo de incógnitas
            seq_iter[-ahead + h, 0] = yh
        return np.array(preds, dtype=float)

    # Casos raros: ajustar tamaño
    if y0.size > ahead:
        return y0[-ahead:]
    return np.pad(y0, (0, ahead - y0.size), mode='edge')

def predict_for_model_ahead(model, mtype, X_exog, y_scaled, scaler_y,
                            start_index, seq_len, ahead, num_points):
    """
    Multi-ahead para un 'ahead' fijo. Devuelve: reales_concat, directa_concat, autoreg_concat
    """
    n = X_exog.shape[0]
    num_blocks = max(0, num_points // ahead)

    reales_blocks = []
    directa_blocks = []
    autoreg_blocks = []

    auto_scaled_flat = np.array([], dtype=float)  # buffer en ESCALA (plano)

    for k in range(num_blocks):
        start_idx = start_index + ahead * k
        end_idx   = start_idx + seq_len + ahead
        if end_idx > n:
            break

        # ====== Secuencia DIRECTA ======
        seq_dir = np.concatenate(
            [X_exog[start_idx:end_idx, 0:1], X_exog[start_idx:end_idx, 3:]],
            axis=1
        ).copy()
        seq_dir[-ahead:, 0] = 0  # incógnitas

        pred_scaled_direct = predict_block_scaled(model, mtype, seq_dir, seq_len, ahead)  # (ahead,)
        pred_direct_real   = inverse_block(scaler_y, pred_scaled_direct)

        # Reales del mismo tramo
        reales_block = scaler_y.inverse_transform(y_scaled[end_idx - ahead:end_idx])[:, 0]

        directa_blocks.append(pred_direct_real)
        reales_blocks.append(reales_block)

        # ====== Secuencia AUTORREGRESIVA ======
        seq_auto = np.concatenate(
            [X_exog[start_idx:end_idx, 0:1], X_exog[start_idx:end_idx, 3:]],
            axis=1
        ).copy()

        # Realimenta con los j últimos valores predichos (buffer plano entre bloques)
        max_back = min(k, seq_len)
        for j in range(1, max_back + 1):
            if auto_scaled_flat.size >= j:
                seq_auto[-(j + ahead), 0] = float(auto_scaled_flat[-j])

        seq_auto[-ahead:, 0] = 0  # incógnitas

        pred_scaled_auto = predict_block_scaled(model, mtype, seq_auto, seq_len, ahead)   # (ahead,)
        auto_scaled_flat = np.hstack([auto_scaled_flat, pred_scaled_auto]) if auto_scaled_flat.size else pred_scaled_auto

        pred_auto_real   = inverse_block(scaler_y, pred_scaled_auto)
        autoreg_blocks.append(pred_auto_real)

    # Concatena todos los bloques
    reales_concat   = np.concatenate(reales_blocks)  if reales_blocks  else np.array([])
    directa_concat  = np.concatenate(directa_blocks) if directa_blocks else np.array([])
    autoreg_concat  = np.concatenate(autoreg_blocks) if autoreg_blocks else np.array([])

    # recorta a 'num_points'
    reales_concat  = reales_concat[:num_points]
    directa_concat = directa_concat[:num_points]
    autoreg_concat = autoreg_concat[:num_points]

    return reales_concat, directa_concat, autoreg_concat

def main():
    # -------- Carga datos y scalers --------
    print("Cargando dataset y preprocesando...")
    df = pd.read_csv(DATA_CSV)
    X_exog, y_scaled, _, _ = preprocess_data_electricity(df)

    print("Cargando scalers...")
    scaler_X = joblib.load(os.path.join(MODELS_DIR, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(MODELS_DIR, "scaler_y.pkl"))

    # -------- Descubre modelos para el AHEAD deseado --------
    model_files = find_model_files(MODELS_DIR, NumAhead)
    if not model_files:
        raise RuntimeError(f"No se encontraron modelos que terminen en '_{NumAhead}' en {MODELS_DIR}")

    print("Modelos detectados (ahead =", NumAhead, "):")
    for f in model_files:
        print("  -", os.path.basename(f))

    # -------- Ejecuta predicciones --------
    resultados = {}          # nombre_modelo -> dict con arrays
    reales_ref = None

    for path in model_files:
        model, mtype = load_any_model(path)
        name = os.path.splitext(os.path.basename(path))[0]

        print(f"\nProcesando {name} ({mtype}) ...")
        reales, pdir, paut = predict_for_model_ahead(
            model, mtype, X_exog, y_scaled, scaler_y,
            start_index=indice_inicial,
            seq_len=SEQ_LENGTH,
            ahead=NumAhead,
            num_points=NUM_PUNTOS
        )

        # Debug útil (puedes dejarlo o quitarlo):
        print(f"  tamaños -> reales:{len(reales)} directa:{len(pdir)} autoreg:{len(paut)}")

        if reales_ref is None:
            reales_ref = reales

        # iguala longitudes a la mínima por seguridad en ESTA iteración
        L = min(len(reales_ref), len(reales), len(pdir), len(paut))
        reales_ref = reales_ref[:L]
        resultados[name] = {
            "directa": pdir[:L],
            "autoregresiva": paut[:L]
        }

    # ---------- Alineación FINAL ----------
    if resultados:
        final_L = len(reales_ref)
        for v in resultados.values():
            final_L = min(final_L, len(v["directa"]), len(v["autoregresiva"]))
        reales_ref = reales_ref[:final_L]
        for k in list(resultados.keys()):
            resultados[k]["directa"]       = resultados[k]["directa"][:final_L]
            resultados[k]["autoregresiva"] = resultados[k]["autoregresiva"][:final_L]

    pasos = np.arange(len(reales_ref))

    # -------- Métricas (MAE y RMSE) --------
    print(f"\nEvaluación {DIA:02d}/{MES:02d}/{ANIO} - Prueba Nº {NumPrueba} para Ahead={NumAhead}")
    print("\nMAE (Directa / Autoregresiva):")
    for k, v in resultados.items():
        mae_dir = np.mean(np.abs(reales_ref - v["directa"]))
        mae_aut = np.mean(np.abs(reales_ref - v["autoregresiva"]))
        print(f"  {k:35s}  {mae_dir:8.3f} / {mae_aut:8.3f}")

    print("\nRMSE (Directa / Autoregresiva):")
    for k, v in resultados.items():
        rmse_dir = np.sqrt(np.mean((reales_ref - v["directa"])**2))
        rmse_aut = np.sqrt(np.mean((reales_ref - v["autoregresiva"])**2))
        print(f"  {k:35s}  {rmse_dir:8.3f} / {rmse_aut:8.3f}")
    print("\n")

    # -------- Gráficas --------

    # # Primera gráfica directa, donde muestro el resultado de los 6 primeros de los 12 modelos 
    # plt.figure(figsize=(12, 7))
    # plt.plot(pasos, reales_ref, label="Actual", linewidth=1.6, color='tab:blue')
    # for k, v in resultados.items():
    #     plt.plot(pasos, v["directa"], linestyle=(0, (5, 3)), label=f"{k} (Direct)")
    # plt.title(f"Consumption forecast: Actual & direct predictions - Start {DIA:02d}/{MES:02d}/{ANIO} | ahead={NumAhead}")
    # plt.ylabel("Energy [kWh]"); plt.xlabel("10-min step"); plt.grid(True); plt.legend(ncol=2, fontsize=8)
    # plt.tight_layout(); plt.show()
    if ploteoGraficas==True:
        # Obtener lista de keys ordenados (Python 3.7+ mantiene el orden de inserción)
        model_keys = list(resultados.keys())
        first_six = model_keys[:6]
        last_six = model_keys[6:]

        # Primera gráfica direct: primeros 6 modelos
        plt.figure(figsize=(12, 7))
        plt.plot(pasos, reales_ref, label="Actual", linewidth=1.6, color='tab:blue')
        for k in first_six:
            plt.plot(pasos, resultados[k]["directa"], linestyle=(0, (5, 3)), label=f"{k} (Direct)")
        plt.title(f"Consumption forecast: Actual and direct predictions - Start {DIA:02d}/{MES:02d}/{ANIO} | ahead={NumAhead}")
        plt.ylabel("Energy [kWh]")
        plt.xlabel("10-min step")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

        # Segunda gráfica directo: últimos 6 modelos
        plt.figure(figsize=(12, 7))
        plt.plot(pasos, reales_ref, label="Actual", linewidth=1.6, color='tab:blue')
        for k in last_six:
            plt.plot(pasos, resultados[k]["directa"], linestyle=(0, (5, 3)), label=f"{k} (Direct)")

        plt.title(f"Consumption forecast: Actual and direct predictions - Start {DIA:02d}/{MES:02d}/{ANIO} | ahead={NumAhead}")
        plt.ylabel("Energy [kWh]")
        plt.xlabel("10-min step")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()  

        # plt.figure(figsize=(12, 7))
        # plt.plot(pasos, reales_ref, label="Actual", linewidth=1.6)
        # for k, v in resultados.items():
        #     plt.plot(pasos, v["autoregresiva"], linestyle=(0, (5, 3)), label=f"{k} (Autoreg)")
        # plt.title(f"Consumption forecast: Actual & autoregressive - Start {DIA:02d}/{MES:02d}/{ANIO} | ahead={NumAhead}")
        # plt.ylabel("Energy [kWh]"); plt.xlabel("10-min step"); plt.grid(True); plt.legend(ncol=2, fontsize=8)
        # plt.tight_layout(); plt.show()

        # Primera gráfica autoregresiva, donde muestro el resultado de los 6 primeros de los 12 modelos
        plt.figure(figsize=(12, 7))
        plt.plot(pasos, reales_ref, label="Actual", linewidth=1.6, color='tab:blue')
        for k in first_six:
            plt.plot(pasos, resultados[k]["autoregresiva"], linestyle=(0, (5, 3)), label=f"{k} (Autoreg)")
        plt.title(f"Consumption forecast: Actual and autoregressive predictions - Start {DIA:02d}/{MES:02d}/{ANIO} | ahead={NumAhead}")
        plt.ylabel("Energy [kWh]")
        plt.xlabel("10-min step")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()  

        # Segunda gráfica autoregresiva, donde muestro el resultado de los 6 últimos de los 12 modelos
        plt.figure(figsize=(12, 7))
        plt.plot(pasos, reales_ref, label="Actual", linewidth=1.6, color='tab:blue')
        for k in last_six:
            plt.plot(pasos, resultados[k]["autoregresiva"], linestyle=(0, (5, 3)), label=f"{k} (Autoreg)")
        plt.title(f"Consumption forecast: Actual and autoregressive predictions - Start {DIA:02d}/{MES:02d}/{ANIO} | ahead={NumAhead}")
        plt.ylabel("Energy [kWh]")
        plt.xlabel("10-min step")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
