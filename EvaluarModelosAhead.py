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

# Cargar el modelo y los escaladores
#model = load_model('models/modelo_LSTM_numlayers2.h5')
model = load_model('models/LSTM_89_1.h5', custom_objects={'mse': tf.keras.losses.mse})

scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')

# Cargar y preprocesar el dataset
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y_scaled, _, _ = preprocess_data_electricity(df)

SEQ_LENGTH = 48  # igual que en entrenamiento
num_predicciones = 6*24  # número de predicciones que quieres realizar
indice_inicial = 1000  # por ejemplo, desde la posición 1000

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
    entrada_directa = secuencia_directa.reshape(1, SEQ_LENGTH+1, secuencia_directa.shape[1])
    pred_esc_directa = model.predict(entrada_directa)
    pred_real_directa = scaler_y.inverse_transform(
        np.hstack((pred_esc_directa, np.zeros((pred_esc_directa.shape[0], 2))))
    )[:, 0][0]
    valor_real = scaler_y.inverse_transform(
        y_scaled[end_idx].reshape(1, -1)
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
    entrada_auto = secuencia_auto.reshape(1, SEQ_LENGTH+1, secuencia_auto.shape[1])
    pred_esc_auto = model.predict(entrada_auto)
    pred_real_auto = scaler_y.inverse_transform(
        np.hstack((pred_esc_auto, np.zeros((pred_esc_auto.shape[0], 2))))
    )[:, 0][0]

    valores_predichos_autoregresivo_ne.append(pred_esc_auto)
    valores_predichos_autoregresivo.append(pred_real_auto)
# ------- GRÁFICA -------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(range(num_predicciones), valores_reales, label='Real')
axes[0].plot(range(num_predicciones), valores_predichos_directo, label='Predicción directa', linestyle='--')
axes[0].set_title('Real vs Predicción directa (siempre reales como entrada)')
axes[0].legend()
axes[0].grid()

axes[1].plot(range(num_predicciones), valores_reales, label='Real')
axes[1].plot(range(num_predicciones), valores_predichos_autoregresivo, label='Predicción autoregresiva', linestyle='--')
axes[1].set_title('Real vs Predicción autoregresiva (acumulando predicciones)')
axes[1].legend()
axes[1].grid()

plt.xlabel('Step')
plt.tight_layout()
plt.show()
