#-----------------------------------
#Con este script puedes evaluar UN SOLO modelo entrenado
#para diferentes pasos adelante (num_ahead)
#-----------------------------------

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use 'Agg' if you do not require a GUI
import matplotlib.pyplot as plt
import types  # para enlazar el método correctamente
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import defaultdict
from itertools import cycle
from scripts.preprocess_data_Tetuan import preprocess_data_electricity
import plotly.express as px
from scripts.buscarIndexTetuan import fila_para_fecha
matplotlib.rcParams['savefig.directory'] = r'/mnt/c/Users/Sebas/Desktop/PowerML/output/ImagesVarias/cap4'  # p.ej. "C:\\Users\\yo\\figs" en Windows
matplotlib.rcParams['savefig.format'] = 'png'                      # extensión por defecto
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------- CONFIGURACIÓN ---------- (Valores fijos    )
SEQ_LENGTH  = 48          # 8 h de pasado
num_ahead = [1,6,12,24,48]  # número de pasos adelante que quieres predecir
num_ahead =[1] 

MODELS_DIR  = 'models'    # ahí están los .h5     (p.e. LSTM_89_1.h5)
SCALER_DIR  = 'models'    # ahí están los scaler_X.pkl y scaler_y.pkl

Tipo = 'LSTMCNN'            # 'LSTM' o 'LSTMCNN'
numModel = 43

# Del 1 al 4
Grafica = 1

NUM_DIAS = 7
Variante = str(numModel)          # 'Bi' o 'Uni'

# ---------- FECHA DE ARRANQUE DESEADA ----------
if Grafica in [1,2]:
    DIA, MES, ANIO = 6, 3, 2017        # 01-01-2017 00:00 a 30-12-2017 23:50

if Grafica in [3,4]:
    DIA, MES, ANIO = 12, 7, 2017        # 01-01-2017 00:00 a 30-12-2017 23:50    

HoraINI = "00:00"
indice_inicial = fila_para_fecha(DIA, MES, ANIO)
num_predicciones_max   = 6 * 24 * NUM_DIAS  # puntos que quieres pintar (6/h , 24h/dia)

if Grafica in [2,4]:
    indice_inicial += 6*12  # offset para probar otro punto desde el medio dia
    HoraINI = "12:00"

def _default_name(self):
    return f"{Tipo}_{numModel}_{Grafica}.png"   # pon aquí tu nombre por defecto


# num_predicciones_max = 6

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
    model = load_model(os.path.join(MODELS_DIR, f'{Tipo}_{Variante}_{kk}.h5'),
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

# Primero, calcular los valores de las predicciones agregadas para cada método
legendas = ["10 minutes", "1 hour", "2 hours", "4 hours", "8 hours"]

# Calcular los valores de la predicción directa
real_directo_values = None
pred_directo_values = {}
for kk, datos in resultados.items():
    real_tmp = np.concatenate(datos["directo"]["real"])
    pred_tmp = np.concatenate(datos["directo"]["pred"])
    pred_directo_values[kk] = pred_tmp
    if real_directo_values is None:
        real_directo_values = real_tmp

# Calcular los valores de la predicción autoregresiva
real_auto_values = None
pred_auto_values = {}
for kk, datos in resultados.items():
    real_tmp = np.concatenate(datos["auto"]["real"])
    pred_tmp = np.concatenate(datos["auto"]["pred"])
    pred_auto_values[kk] = pred_tmp
    if real_auto_values is None:
        real_auto_values = real_tmp

# ---------- PRIMERA GRÁFICA: Dos subplots ----------
if 0:   
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot para predicción directa
    pasos_directo = np.arange(len(real_directo_values))
    axes[0].plot(pasos_directo, real_directo_values, linestyle='-', label='Real')
    for kk, pred in pred_directo_values.items():
        axes[0].plot(pasos_directo, pred, linestyle='--', label=f'Pred. Directa {legendas[kk]}')
    axes[0].set_title('Real vs Predicción directa')
    axes[0].legend()
    axes[0].grid()

    # Subplot para predicción autoregresiva
    pasos_auto = np.arange(len(real_auto_values))
    axes[1].plot(pasos_auto, real_auto_values, linestyle='-', label='Real')
    for kk, pred in pred_auto_values.items():
        axes[1].plot(pasos_auto, pred, linestyle='--', label=f'Pred. Autoreg. {legendas[kk]}')
    axes[1].set_title('Real vs Predicción autoregresiva')
    axes[1].set_xlabel('Step')
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()

    # ---------- SEGUNDA GRÁFICA: Sobreposición de ambas predicciones en una misma figura ----------
fig2, ax2 = plt.subplots(figsize=(12, 6))
pasos = np.arange(len(real_directo_values))
ax2.plot(pasos, real_directo_values, label='Actual', linewidth=1.6, color='tab:blue')
for kk, pred in pred_directo_values.items():
    ax2.plot(pasos, pred, linestyle=(0, (5, 3)), color='tab:red', linewidth=1 , label=f'Direct-{legendas[kk]} ahead')
for kk, pred in pred_auto_values.items():
    ax2.plot(pasos, pred, linestyle=(0, (5, 3)), color='tab:green', linewidth=1, label=f'Autoregressive-{legendas[kk]} ahead')

fig2.canvas.get_default_filename = types.MethodType(_default_name, fig2.canvas) # enlaza el método para guardar con el nombre por defecto
ax2.set_title(f"Consumption forecast: Actual series and predictions - Start {DIA:02d}/{MES:02d}/{ANIO} at {HoraINI}")
ax2.set_ylabel("Energy [kWh]")
ax2.set_xlabel("10-min step")
ax2.legend(ncol=2)
ax2.grid(True)
plt.tight_layout()
plt.show()
aa=1

error_medios = {}

for flag, datos in resultados.items():
    # Consolidate arrays (assuming each iteration corresponds to one prediction)
    real_directo = np.concatenate(datos["directo"]["real"])
    pred_directo = np.concatenate(datos["directo"]["pred"])
    mae_directo = np.mean(np.abs(real_directo - pred_directo))
    mse_directo = np.mean((real_directo - pred_directo)**2)
    rmse_directo = np.sqrt(mse_directo)
    
    real_auto = np.concatenate(datos["auto"]["real"])
    pred_auto = np.concatenate(datos["auto"]["pred"])
    mae_auto = np.mean(np.abs(real_auto - pred_auto))
    mse_auto = np.mean((real_auto - pred_auto)**2)
    rmse_auto = np.sqrt(mse_auto)
    
    error_medios[flag] = {
        "mse_directo": mse_directo,
        "rmse_directo": rmse_directo,
        "mae_directo": mae_directo,
        "mae_auto": mae_auto,
        "mse_auto": mse_auto,
        "rmse_auto": rmse_auto
    }
# Imprimir los errores medios de cada modelo
# Print the error metrics for each model
for flag, e in error_medios.items():
    print(f"Modelo {Tipo}_{numModel}_{Grafica}:")
    print(f"   RMSE Directo = {e['rmse_directo']:.2f}, MAE Directo = {e['mae_directo']:.2f}")
    print(f"   RMSE Autoregresivo = {e['rmse_auto']:.2f}, MAE Autoregresivo = {e['mae_auto']:.2f}")
    print("-------------------------------")
    print("")
    print("")

