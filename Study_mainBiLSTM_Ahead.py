import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dropout, Dense, Input, Bidirectional, Multiply, Flatten)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from time import time

# --- Import de tu pipeline ---
from scripts.preprocess_data_Tetuan import preprocess_data_electricity

# --------------- CONFIGURACIÓN HIPERPARÁMETROS ----------------
# Puedes editar libremente estos hiperparámetros (modelo y entrenamiento)
SEQ_LENGTH = 48                              # Ventana base de entrada (p.ej., 8h si muestras cada 10 min)
PREDICTION_HORIZONS = [1, 6, 12, 24, 48]     # 10 min, 1h, 2h, 4h, 8h

NUM_CONFIGURACION_ESTUDIO = 138

# (Entrenamiento) — ajusta tú los valores
EPOCHS = 50
BATCH_SIZE = 32

# (Arquitectura BiLSTM) — ajusta tú los valores
UNITS_LSTM1 = 64        # tamaño de la 1ª BiLSTM
UNITS_LSTM2 = 32         # tamaño de la 2ª BiLSTM
DROPOUT_RATE = 0.2       # 0.0 (sin dropout) … 0.5
USE_FEATURE_ATT = True  # atención sobre features
USE_TEMPORAL_ATT = False # atención sobre el eje temporal
DENSE_HIDDEN = 1024      # tamaño de la capa densa previa a la salida
LEARNING_RATE = 0.001
# ------------------ CARGA Y PREPROCESADO ------------------
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)

# ------------------ CREACIÓN DE SECUENCIAS MULTI-STEP ------------------
def create_multistep_sequences(data, seq_length, pred_length):
    X, Y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        seq = np.concatenate([data[i:i + seq_length + pred_length, 0:1],data[i:i + seq_length + pred_length, 3:]], axis=1).copy()
        seq[-pred_length:, 0] = 0  # enmascara el target futuro
        X.append(seq)
        Y.append(data[i + seq_length:i + seq_length + pred_length, 0])
    return np.array(X), np.array(Y)

# ------------------ CAPAS DE ATENCIÓN ------------------
def feature_attention(inputs):
    """Atención de características: aprende pesos por canal (feature)."""
    attn = Dense(inputs.shape[-1], activation='sigmoid')(inputs)
    attn = Dense(inputs.shape[-1], activation='softmax')(attn)
    return Multiply()([inputs, attn])

def temporal_attention(inputs):
    seq_length = inputs.shape[1]
    attention = Dense(1, activation='relu')(inputs)
    attention = Flatten()(attention)
    attention = Dense(seq_length, activation='softmax')(attention)
    attention = tf.keras.layers.Reshape((seq_length, 1))(attention)
    return Multiply()([inputs, attention])

# ------------------ MODELO BiLSTM AHEAD ------------------
def build_bilstm_ahead(total_seq_len, num_features, pred_length,units_lstm1=UNITS_LSTM1, units_lstm2=UNITS_LSTM2,dropout_rate=DROPOUT_RATE,use_feature_att=USE_FEATURE_ATT, use_temporal_att=USE_TEMPORAL_ATT,dense_hidden=DENSE_HIDDEN):
    inputs = Input(shape=(total_seq_len, num_features))
    bi_lstm = Bidirectional(LSTM(units_lstm1, return_sequences=True))(inputs)
    if dropout_rate and dropout_rate > 0:
        bi_lstm = Dropout(dropout_rate)(bi_lstm)

    if use_feature_att:
        bi_lstm = feature_attention(bi_lstm)

    bi_lstm2 = Bidirectional(LSTM(units_lstm2, return_sequences=True))(bi_lstm)

    if use_temporal_att:
        bi_lstm2 = temporal_attention(bi_lstm2)

    flattened = Flatten()(bi_lstm2)
    dense1 = Dense(dense_hidden, activation='relu')(flattened)
    outputs = Dense(pred_length)(dense1)   # <<— salida multi-step

    return Model(inputs, outputs)
# =======================
#      BUCLE DE ESTUDIO
# =======================
results = []

for pred_len in PREDICTION_HORIZONS:
    print(f"Entrenando BiLSTM para predicción a {pred_len} pasos...")

    # Construir secuencias con (SEQ_LENGTH + pred_len) timesteps
    X_seq, y_seq = create_multistep_sequences(X_exog, SEQ_LENGTH, pred_len)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    num_features = X_seq.shape[2]
    total_timesteps = SEQ_LENGTH + pred_len

    # Modelo
    model = build_bilstm_ahead(total_seq_len=total_timesteps,
                               num_features=num_features,
                               pred_length=pred_len)

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Entrenamiento
    start_time = time()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=0
    )
    training_time = time() - start_time

    # Predicción
    predictions = model.predict(X_test, verbose=0)

    # Desnormalización para métricas agregadas
    y_test_flat = y_test.flatten().reshape(-1, 1)
    pred_flat = predictions.flatten().reshape(-1, 1)

    y_test_real = scaler_y.inverse_transform(
        np.hstack((y_test_flat, np.zeros((len(y_test_flat), 2))))
    )[:, 0]
    predictions_real = scaler_y.inverse_transform(
        np.hstack((pred_flat, np.zeros((len(pred_flat), 2))))
    )[:, 0]

    # Métricas
    mse = mean_squared_error(y_test_real, predictions_real)
    r2 = r2_score(y_test_real, predictions_real)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_real - predictions_real) / (y_test_real + 1e-8))) * 100
    mae = np.mean(np.abs(y_test_real - predictions_real))
    mre = np.mean(np.abs((y_test_real - predictions_real) / (y_test_real + 1e-8))) * 100
    cvrmse = np.std((y_test_real - predictions_real) / (y_test_real + 1e-8)) * 100

    train_loss_final = history.history['loss'][-1]
    val_loss_final = history.history['val_loss'][-1]
    overfit_gap = val_loss_final - train_loss_final
    loss_ratio = (val_loss_final / train_loss_final) if train_loss_final != 0 else np.nan

    results.append({
        'modelFamily': 'BiLSTM',
        'prediction_horizon': pred_len,
        'UNITS_LSTM1': UNITS_LSTM1,
        'UNITS_LSTM2': UNITS_LSTM2,
        'DROPOUT_RATE': DROPOUT_RATE,
        'FEATURE_ATT': USE_FEATURE_ATT,
        'TEMPORAL_ATT': USE_TEMPORAL_ATT,
        'DENSE_HIDDEN': DENSE_HIDDEN,
        'EPOCHS': EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LR': LEARNING_RATE,
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

    # Guardar modelo (por horizonte)
    model.save(f"models/BiLSTM_ahead_{NUM_CONFIGURACION_ESTUDIO}_{pred_len}.h5")

    # Guardar resultados incrementales
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"./output/AheadDL/BiLSTM_multistep_prediction_results_{NUM_CONFIGURACION_ESTUDIO}.csv", index=False)

print("Estudio BiLSTM-ahead completado.")
