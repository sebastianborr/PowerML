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
model = load_model('models/modelo_LSTM_numlayers1.h5', custom_objects={'mse': tf.keras.losses.mse})

scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')

# Cargar y preprocesar el dataset
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y_scaled, _, _ = preprocess_data_electricity(df)

SEQ_LENGTH = 48  # igual que en entrenamiento
num_predicciones = 6*12  # número de predicciones que quieres realizar
indice_inicial = 1000  # por ejemplo, desde la posición 1000

valores_reales = []
valores_predichos = []

for k in range(num_predicciones):
    # Tomar la secuencia de entrada para la posición deseada
    start_idx = indice_inicial + k
    end_idx = start_idx + SEQ_LENGTH
    if end_idx >= len(X_exog):  # Evita pasarte del final
        break

    #secuencia = X_exog[start_idx:end_idx].copy()
    secuencia = np.concatenate([X_exog[start_idx:start_idx + SEQ_LENGTH, 0:1],
                          X_exog[start_idx:start_idx + SEQ_LENGTH, 3:]], axis=1).copy()
   
    secuencia[-1, 0] = 0  # variable objetivo desconocida en la última posición
    #entrada = secuencia.reshape(1, SEQ_LENGTH, X_exog.shape[1])
    entrada = secuencia.reshape(1, SEQ_LENGTH, secuencia.shape[1])
    pred_esc = model.predict(entrada)
    pred_real = scaler_y.inverse_transform(
        np.hstack((pred_esc, np.zeros((pred_esc.shape[0], 2))))
    )[:, 0][0]

    # Valor real de la posición a predecir
    valor_real = scaler_y.inverse_transform(y_scaled[end_idx].reshape(1, -1))[0, 0]

    valores_predichos.append(pred_real)
    valores_reales.append(valor_real)

# Graficar resultados
plt.figure(figsize=(10, 5))
plt.plot(range(num_predicciones), valores_reales, label='Real', marker='.')
plt.plot(range(num_predicciones), valores_predichos, label='Predicción', marker='.', linestyle='--')
plt.xlabel('Step')
plt.ylabel('Consumo kWh')
plt.title('Real vs Predicho')
plt.legend()
plt.grid()
#plt.savefig('grafica.png')
plt.show()
