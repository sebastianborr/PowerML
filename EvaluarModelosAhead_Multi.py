import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use 'Agg' if you do not require a GUI
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import defaultdict
from itertools import cycle
from scripts.preprocess_data_Tetuan import preprocess_data_electricity
import plotly.express as px

# ---------- CONFIGURACIÓN ----------
SEQ_LENGTH  = 48          # 8 h de pasado
num_predicciones_max   = 6 * 24    # puntos que quieres pintar (~1 día)
START_INDEX = 1200
MODELS_DIR  = 'models'    # ahí están los .h5     (p.e. LSTM_89_1.h5)
SCALER_DIR  = 'models'    # ahí están los scaler_X.pkl y scaler_y.pkl
indice_inicial = 17858  # por ejemplo, desde la posición 1000
num_ahead = [1,6,12,24,48]  # número de pasos adelante que quieres predecir
#num_ahead = [1]
# ---------- CARGA DATASET Y SCALERS ----------
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y_scaled, _, _ = preprocess_data_electricity(df)

# Cargar el modelo y los escaladores
#model = load_model('models/modelo_LSTM_numlayers2.h5')

scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')


# ---------- PARÁMETROS ----------
valores_reales = []
valores_predichos_directo = []
valores_predichos_autoregresivo = []
valores_predichos_autoregresivo_ne = []
# Diccionario para almacenar resultados de cada modelo (kk)
resultados = {}
flag = -1
for kk in num_ahead:
    # Inicializamos para cada modelo kk
    flag = flag+1
    resultados[flag] = {
        "directo": {"pred": [], "real": []},
        "auto": {"pred": [],"pred_ne": [], "real": []}
    }
    model = load_model(os.path.join(MODELS_DIR, f'LSTM_89_{kk}.h5'),
                       custom_objects={'mse': tf.keras.losses.mse})
    num_predicciones = num_predicciones_max // kk  # Número de predicciones a realizar para este modelo
    for k in range(num_predicciones):
        start_idx = indice_inicial + kk*k
        end_idx = start_idx + SEQ_LENGTH+ kk
        #if target_end > len(X_exog):
        #    break

        # Predicción directa
        secuencia_directa = np.concatenate([
            X_exog[start_idx:end_idx, 0:1],
            X_exog[start_idx:end_idx, 3:]
        ], axis=1).copy()
        secuencia_directa[-kk:, 0] = 0  # Últimos kk como incógnita
        entrada_directa = secuencia_directa.reshape(1, SEQ_LENGTH+kk, secuencia_directa.shape[1])

        pred_esc_directa = model.predict(entrada_directa)  # Shape (1, kk)

    #    pred_real_directa = scaler_y.inverse_transform(
    #        np.hstack((pred_esc_directa.T, np.zeros((pred_esc_directa.shape[1], 2))))
    #    )[:, :kk].flatten()
#
    #    valor_real_directo = scaler_y.inverse_transform(
    #        y_scaled[end_idx:target_end].reshape(1, -1)
    #    )[:, :kk].flatten()

                # ---------- PREDICCIÓN DIRECTA ----------
        pred_block = pred_esc_directa.flatten()           # (kk,)
        pred_mat    = np.zeros((kk, 3))                   # (kk,3)
        pred_mat[:, 0] = pred_block                       # solo la Zona 1

        pred_real_directa = scaler_y.inverse_transform(pred_mat)[:, 0]   # (kk,)

        valor_real_directo = scaler_y.inverse_transform(
                                y_scaled[end_idx-kk:end_idx]           # (kk,3)  ¡ya correcto!
                            )[:, 0]      
        
        resultados[flag]["directo"]["pred"].append(pred_real_directa)
        resultados[flag]["directo"]["real"].append(valor_real_directo)

        # Predicción autorregresiva
        secuencia_auto = np.concatenate([
            X_exog[start_idx:end_idx, 0:1],
            X_exog[start_idx:end_idx, 3:]
        ], axis=1).copy()

        for j in range(1, k+1):
            if j > SEQ_LENGTH:
                break
            #secuencia_auto[-(j+1), 0] = resultados[kk]["auto"]["pred"][-j]
            secuencia_auto[-(j+kk), 0] = resultados[flag]["auto"]["pred_ne"][-j]  # Usar la última predicción de la iteración anterior

        secuencia_auto[-kk:, 0] = 0  # Últimos kk como incógnita

        entrada_auto = secuencia_auto.reshape(1, SEQ_LENGTH+kk, secuencia_auto.shape[1])

        pred_esc_auto = model.predict(entrada_auto)

        pred_flat = pred_esc_auto.flatten()  # Esto da un vector de shape (kk,)
        if not isinstance(resultados[flag]["auto"]["pred_ne"], np.ndarray) or resultados[flag]["auto"]["pred_ne"].size == 0:
            resultados[flag]["auto"]["pred_ne"] = pred_flat
        else:
            resultados[flag]["auto"]["pred_ne"] = np.hstack((resultados[flag]["auto"]["pred_ne"], pred_flat))


        #resultados[flag]["auto"]["pred_ne"].append(pred_esc_auto)  # Guardar la predicción sin desnormalizar
       
       # pred_real_auto = scaler_y.inverse_transform(
       #     np.hstack((pred_esc_auto.T, np.zeros((pred_esc_auto.shape[1], 2))))
       # )[:, :kk].flatten()
        
        pred_block = pred_esc_auto.flatten()
        pred_mat   = np.zeros((kk, 3))
        pred_mat[:, 0] = pred_block
        pred_real_auto = scaler_y.inverse_transform(pred_mat)[:, 0]
 
        # Usamos los mismos valores reales para comparar que en la directa
        resultados[flag]["auto"]["pred"].append(pred_real_auto)
        resultados[flag]["auto"]["real"].append(valor_real_directo)
# ------- GRÁFICA -------
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Subplot para predicción directa
control = 0
legendas = [" 10 minutos", " 1 hora", " 2 horas", " 4 horas", " 8 horas"]
for kk, datos in resultados.items():
    real_directo = np.array(datos["directo"]["real"]).flatten()
    pred_directo = np.array(datos["directo"]["pred"]).flatten()
    pasos = np.arange(len(real_directo))
    if control ==0:
        control = 1
        axes[0].plot(pasos, real_directo, linestyle='-', label=f'Real')
    axes[0].plot(pasos, pred_directo, linestyle='--', label=f'Pred. Directa {legendas[kk]}')

axes[0].set_title('Real vs Predicción directa (siempre reales como entrada)')
axes[0].legend()
axes[0].grid()

# Subplot para predicción autorregresiva
control = 0
for kk, datos in resultados.items():
    
    real_auto = np.array(datos["auto"]["real"]).flatten()
    pred_auto = np.array(datos["auto"]["pred"]).flatten()
    pasos = np.arange(len(real_auto))
    if control == 0:
        control = 1
        axes[1].plot(pasos, real_auto, linestyle='-', label=f'Real')
    axes[1].plot(pasos, pred_auto, linestyle='--', label= f'Pred. Autoreg. {legendas[kk]}')

axes[1].set_title('Real vs Predicción autoregresiva (acumulando predicciones)')
axes[1].legend()
axes[1].grid()

axes[1].set_xlabel('Step')
plt.tight_layout()
plt.show()

aa=1