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
# num_ahead =[1,48] 

MODELS_DIR  = 'models'    # ahí están los .h5     (p.e. LSTM_89_1.h5)
SCALER_DIR  = 'models'    # ahí están los scaler_X.pkl y scaler_y.pkl

Tipo = 'BiLSTM'            # 'LSTM' o 'LSTMCNN'
numModel = 19

if Tipo in ['XGBoost','GradientBoosting']:
    DL = False                     # True si es DL (.h5/.keras); False si es ML (.pkl)
else:
    DL = True

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
    if DL==True and Tipo != 'Transformer':
        model = load_model(os.path.join(MODELS_DIR, f'{Tipo}_{Variante}_{kk}.h5'),
                       custom_objects={'mse': tf.keras.losses.mse})
    elif DL==True:
        model = load_model(os.path.join(MODELS_DIR, f'{Tipo}_{Variante}_{kk}.keras'),
                       custom_objects={'mse': tf.keras.losses.mse})
    else:
        # model = joblib.load(os.path.join(MODELS_DIR, f'{Tipo}_{Variante}_{kk}.pkl'))
        model = joblib.load(os.path.join(MODELS_DIR, f'{Tipo}_{Variante}_{kk}.joblib'))


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

        if DL==True:
            entrada_directa = secuencia_directa.reshape(1, SEQ_LENGTH+kk, secuencia_directa.shape[1])
        else:
            entrada_directa = secuencia_directa.reshape(1, -1)

        pred_esc_directa = model.predict(entrada_directa)  # Shape (1, kk)

                # ---------- PREDICCIÓN DIRECTA ----------
        pred_block = pred_esc_directa.flatten()           # (kk,)
        pred_mat    = np.zeros((kk, 3))                   # (kk,3)
        pred_mat[:, 0] = pred_block                       # solo la Zona 1

        pred_real_directa = scaler_y.inverse_transform(pred_mat)[:, 0]   # (kk,)

        valor_real_directo = scaler_y.inverse_transform(
                                y_scaled[end_idx-kk-1:end_idx-1]           # (kk,3)  ¡ya correcto!
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
        if DL==True:
            entrada_auto = secuencia_auto.reshape(1, SEQ_LENGTH+kk, secuencia_auto.shape[1])
        else:
            entrada_auto = secuencia_auto.reshape(1, -1)

        pred_esc_auto = model.predict(entrada_auto)

        pred_flat = pred_esc_auto.flatten()  # Esto da un vector de shape (kk,)
        if not isinstance(resultados[flag]["auto"]["pred_ne"], np.ndarray) or resultados[flag]["auto"]["pred_ne"].size == 0:
            resultados[flag]["auto"]["pred_ne"] = pred_flat
        else:
            resultados[flag]["auto"]["pred_ne"] = np.hstack((resultados[flag]["auto"]["pred_ne"], pred_flat))

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
if 1:   #Un modelo, varias gradicas ahead separadas direct y autoregresiva
    plt.figure(figsize=(12, 7))
    pasos_directo = np.arange(len(real_directo_values))
    plt.plot(pasos_directo, real_directo_values, label='Actual', linewidth=1.6, color='tab:blue')
    colores_directos = ['tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']  # Evitamos el azul
    for i, (kk, pred) in enumerate(pred_directo_values.items()):
        color = colores_directos[i % len(colores_directos)]
        plt.plot(pasos_directo, pred, linestyle='--', linewidth=1, label=f'{legendas[kk]} ahead', color=color)
    plt.title(f"Consumption forecast: Actual series and direct predictions - Start {DIA:02d}/{MES:02d}/{ANIO}")
    plt.xlabel('Step')
    plt.ylabel('Energy [kWh]')
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfica para la predicción autoregresiva
    plt.figure(figsize=(12, 7))
    pasos_auto = np.arange(len(real_auto_values))
    plt.plot(pasos_auto, real_auto_values, label='Actual', linewidth=1.6, color='tab:blue')
    colores_auto = ['tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']  # Puedes usar la misma o variar
    for i, (kk, pred) in enumerate(pred_auto_values.items()):
        color = colores_auto[i % len(colores_auto)]
        plt.plot(pasos_auto, pred, linestyle='--', linewidth=1, label=f'Autoregressive-{legendas[kk]} ahead', color=color)
    plt.plot(pasos_auto, pred, linestyle='--', linewidth=1, label=f'Autoregressive-{legendas[kk]} ahead')
    plt.title(f"Consumption forecast: Actual series and autoregressive predictions - Start {DIA:02d}/{MES:02d}/{ANIO}")
    plt.xlabel('Step')
    plt.ylabel('Energy [kWh]')
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:        
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
    print(f"Direct: RMSE={e['rmse_directo']:.1f}; MAE={e['mae_directo']:.1f} \\ Autoreg.: RMSE = {e['rmse_auto']:.1f}; MAE = {e['mae_auto']:.1f}")
    print("-------------------------------")
    print("")
    print("")

