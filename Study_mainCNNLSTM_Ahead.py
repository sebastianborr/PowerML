import numpy as np
import pandas as pd
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['GLOG_minloglevel'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from time import time
from tensorflow.keras.optimizers import Adam
from scripts.preprocess_data_Tetuan import preprocess_data_electricity
#from scripts.generacionModelosLSTMCNN import seleccionModelosHibridos # Importar la función que genera los modelos LSTM para el estudio de ablación
from scripts.generacionModelosCNNLSTM import seleccionModelosCNN_LSTM
# --------------- CONFIGURACIÓN HIPERPARÁMETROS ----------------
SEQ_LENGTH = 48  # Ventana de entrada (8h)
PREDICTION_HORIZONS = [1, 6, 12, 24, 48]  # pasos a predecir (10 min, 1h, 2h, 4h, 8h)
EPOCHS = 50
BATCH_SIZE = 32
DROPOUT = False
CONFIGURACIONMODELO = 2 # Selección del modelo CNNLSTM (1 a 5)
NUM_CONFIGURACION_ESTUDIO = 35  # Para guardar el modelo con un nombre identificativo
# ------------------ CARGA Y PREPROCESADO ------------------
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)

# ------------------ CREACIÓN DE SECUENCIAS MULTI-STEP ------------------
def create_multistep_sequences(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        seq = np.concatenate([data[i:i + seq_length + pred_length, 0:1], data[i:i + seq_length + pred_length, 3:]], axis=1).copy()
        seq[-pred_length:, 0] = 0  # para evitar que el modelo vea el valor a predecir
        X.append(seq)
        y.append(data[i + seq_length:i + seq_length + pred_length, 0])
    return np.array(X), np.array(y)

# ------------------ MODELO LSTM+CNN AHEAD ------------------
def build_lstmcnn_model(total_seq_len, num_features, pred_length, config=CONFIGURACIONMODELO, dropout=DROPOUT):
    """
    Parte de tu generador 'seleccionModelosHibridos' (LSTM->CNN->...->Dense(1))
    y sustituye la última capa por Dense(pred_length) para emitir un vector multi-step.
    'total_seq_len' = SEQ_LENGTH + pred_length para alinear con las ventanas creadas.
    """
    base = seleccionModelosCNN_LSTM(
        flagModelo=config,
        num_features=num_features,
        SEQ_LENGTH=total_seq_len,
        dropout=dropout
    )
    # Reemplazar la última Dense(1) por Dense(pred_length)
    # (Creamos un nuevo Sequential con las mismas capas menos la última)
    new_layers = list(base.layers[:-1]) + [Dense(pred_length)]
    model = Sequential(new_layers)
    return model

# ------------------ BUCLE DE ESTUDIO ------------------
results = []

for pred_len in PREDICTION_HORIZONS:
    print(f"Entrenando para predicción a {pred_len} pasos...")
    X_seq, y_seq = create_multistep_sequences(X_exog, SEQ_LENGTH, pred_len)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    num_features = X_seq.shape[2]

    total_timesteps = SEQ_LENGTH + pred_len
    # model = build_lstmcnn_model(SEQ_LENGTH, num_features, pred_len)
    model = build_lstmcnn_model(
        total_seq_len=total_timesteps,
        num_features=num_features,
        pred_length=pred_len,
        dropout=DROPOUT
    )
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    start_time = time()
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=0)
    training_time = time() - start_time

    predictions = model.predict(X_test)

    # Desnormalizar
    y_test_flat = y_test.flatten().reshape(-1, 1)
    pred_flat = predictions.flatten().reshape(-1, 1)
    y_test_real = scaler_y.inverse_transform(np.hstack((y_test_flat, np.zeros((len(y_test_flat), 2)))))[:, 0]
    predictions_real = scaler_y.inverse_transform(np.hstack((pred_flat, np.zeros((len(pred_flat), 2)))))[:, 0]

    # Métricas
    mse = mean_squared_error(y_test_real, predictions_real)
    r2 = r2_score(y_test_real, predictions_real)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
    mae = np.mean(np.abs(y_test_real - predictions_real))
    mre = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
    cvrmse = np.std((y_test_real - predictions_real) / y_test_real) * 100

    train_loss_final = history.history['loss'][-1]
    val_loss_final = history.history['val_loss'][-1]
    overfit_gap = val_loss_final - train_loss_final
    loss_ratio = val_loss_final / train_loss_final if train_loss_final != 0 else np.na

    results.append({
        'modelNumber': NUM_CONFIGURACION_ESTUDIO,
        'prediction_horizon': pred_len,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'CVRMSE': cvrmse,
        'Training Time (s)': training_time,
        'Train Loss Final': train_loss_final,
        'Validation Loss Final': val_loss_final,
        'Overfit Gap': overfit_gap,
        'Loss Ratio': loss_ratio
    })
    model.save(f"models/CNNLSTM_{NUM_CONFIGURACION_ESTUDIO}_{pred_len}.h5")

    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"./output/AheadDL/CNNLSTM_multistep_prediction_results_{NUM_CONFIGURACION_ESTUDIO}.csv", index=False)

print("Estudio de predicción a diferentes horizontes completado.")
