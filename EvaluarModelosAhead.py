import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
import plotly.express as px
import joblib
import matplotlib
matplotlib.use('TkAgg')  # O prueba 'Qt5Agg' si no tienes tkinter
import matplotlib.pyplot as plt
from scripts.preprocess_data_Tetuan import preprocess_data_electricity
from scripts.buscarIndexTetuan import fila_para_fecha

# Opcional: cambiar el directorio por defecto para guardar figuras
matplotlib.rcParams['savefig.directory'] = r'/mnt/c/Users/Sebas/Desktop/PowerML'  # p.ej. "C:\\Users\\yo\\figs" en Windows

# Cargar el modelo y los escaladores
model = load_model('models/LSTMCNN_18_1.h5', custom_objects={'mse': tf.keras.losses.mse})
#model = joblib.load('models/ML_RL.pkl')
DL = True  # Indica si el modelo es Deep Learning (LSTM) o ML clásico (RL, RF, etc.)
scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')


# FECHA DE ARRANQUE DESEADA
indice_inicial = 17858  # por ejemplo, desde la posición 1000
    # Otra opción
DIA, MES, ANIO = 15, 4, 2017        # 01-01-2017 00:00 a 30-12-2017 23:50
DIAS_A_PREDECIR = 7
indice_inicial = fila_para_fecha(DIA, MES, ANIO)
num_predicciones   = 6 * 24 * DIAS_A_PREDECIR  # puntos que quieres pintar (6/h , 24h/dia)


# Cargar y preprocesar el dataset
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y_scaled, _, _ = preprocess_data_electricity(df)

SEQ_LENGTH = 48  # igual que en entrenamiento
num_predicciones = 6*24*7 # número de predicciones que quieres realizar

valores_reales = []
valores_predichos_directo = []
valores_predichos_autoregresivo = []
valores_predichos_autoregresivo_ne = []
# ------- PREDICCIÓN DIRECTA (como antes, cada ventana usa solo reales) -------
for k in range(num_predicciones):
    start_idx = indice_inicial + k
    end_idx = start_idx + SEQ_LENGTH+1
    if end_idx >= len(X_exog):
        break

    # PREDICCIÓN DIRECTA: usar la secuencia con todos los datos reales
    secuencia_directa = np.concatenate([
        X_exog[start_idx:end_idx, 0:1],
        X_exog[start_idx:end_idx, 3:]
    ], axis=1).copy()
    secuencia_directa[-1, 0] = 0  # Último como incógnita

    if DL:
        entrada_directa = secuencia_directa.reshape(1, SEQ_LENGTH+1, secuencia_directa.shape[1])
    else:
        entrada_directa = secuencia_directa.reshape(1, -1)
    
    pred_esc_directa = model.predict(entrada_directa)
    if DL:
        pred_real_directa = scaler_y.inverse_transform(
            np.hstack((pred_esc_directa, np.zeros((pred_esc_directa.shape[0], 2))))
        )[:, 0][0]
    else:
        pred_real_directa = scaler_y.inverse_transform(
        np.hstack((pred_esc_directa.reshape(-1, 1), np.zeros((pred_esc_directa.shape[0], 2))))
        )[:, 0][0] 

    valor_real = scaler_y.inverse_transform(
        y_scaled[end_idx-1].reshape(1, -1)      # -1 porque end_idx es exclusivo
    )[0, 0]

    valores_predichos_directo.append(pred_real_directa)
    valores_reales.append(valor_real)

    # PREDICCIÓN AUTORREGRESIVA: crear la secuencia base y reemplazar con predicciones previas
    secuencia_auto = np.concatenate([
        X_exog[start_idx:end_idx, 0:1],
        X_exog[start_idx:end_idx, 3:]
    ], axis=1).copy()

    # Reemplazar los últimos k valores (menos el último) con las predicciones previas
    for j in range(1, k+1):
        if j > SEQ_LENGTH: 
            break
        secuencia_auto[-(j+1), 0] = valores_predichos_autoregresivo_ne[-j]
    
    secuencia_auto[-1, 0] = 0  # Último valor como incógnita
    if DL:
        entrada_auto = secuencia_auto.reshape(1, SEQ_LENGTH+1, secuencia_auto.shape[1])
    else:
        entrada_auto = secuencia_auto.reshape(1, -1)
    #
    pred_esc_auto = model.predict(entrada_auto)
    
    if DL:
        pred_real_auto = scaler_y.inverse_transform(
            np.hstack((pred_esc_auto, np.zeros((pred_esc_auto.shape[0], 2))))
        )[:, 0][0]
    else:
        pred_real_auto = scaler_y.inverse_transform(
            np.hstack((pred_esc_auto.reshape(-1, 1), np.zeros((pred_esc_auto.shape[0], 2))))
        )[:, 0][0]

    valores_predichos_autoregresivo_ne.append(pred_esc_auto)
    valores_predichos_autoregresivo.append(pred_real_auto)
# ------- GRÁFICA -------
import matplotlib.pyplot as plt
if 0:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(range(num_predicciones), valores_reales, label='Real')
    axes[0].plot(range(num_predicciones), valores_predichos_directo, label='Predicción directa', linestyle='--')
    axes[0].set_title(f"Direct prediction from {DIA}/{MES}/{ANIO} - Prediction days: {DIAS_A_PREDECIR:.0f}")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(range(num_predicciones), valores_reales, label='Real')
    axes[1].plot(range(num_predicciones), valores_predichos_autoregresivo, label='Predicción autoregresiva', linestyle='--')
    axes[1].set_title(f"Autoregressive prediction from {DIA}/{MES}/{ANIO} - Prediction days: {DIAS_A_PREDECIR:.0f}")
    axes[1].legend()
    axes[1].grid()
    axes[0].set_ylabel("kWh")
    axes[1].set_ylabel("kWh")

    plt.xlabel('Step')
    plt.tight_layout()
    plt.show()
# ------- GRÁFICA de las tres series juntas (combinadas) -------

fig2, ax2 = plt.subplots(figsize=(10, 6))
pasos = np.arange(len(valores_reales))

ax2.plot(pasos, valores_reales, label='Real', color='tab:blue')
ax2.plot(pasos, valores_predichos_directo, label='Direct Prediction', linestyle='--', color='tab:red')

ax2.set_title(f"Prediction from {DIA}/{MES}/{ANIO} - Prediction days: {DIAS_A_PREDECIR:.0f}")
ax2.set_ylabel("kWh")
ax2.set_xlabel('Step (10 min each)')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()


fig3, ax3 = plt.subplots(figsize=(10, 6))
pasos = np.arange(len(valores_reales))

ax3.plot(pasos, valores_reales, label='Real', color='tab:blue')
ax3.plot(pasos, valores_predichos_directo, label='Direct Prediction', linestyle='--', color='tab:red')
ax3.plot(pasos, valores_predichos_autoregresivo, label='Autoregressive prediction', linestyle='--', color='tab:green')

ax3.set_title(f"Prediction from {DIA}/{MES}/{ANIO} - Prediction days: {DIAS_A_PREDECIR:.0f}")
ax3.set_ylabel("kWh")
ax3.set_xlabel('Step (10 min each)')
ax3.legend()
ax3.grid(True)
plt.tight_layout()
plt.show()


