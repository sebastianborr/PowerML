import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, RepeatVector, TimeDistributed, Flatten
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from time import time

# =========================
#   CONFIGURACIÓN (EDITAR)
# =========================
# Ventana de entrada (en pasos de 10 min; 48 = 8 horas)
SEQ_LENGTH = 48
NUM_CONFIGURACION_ESTUDIO = 123  # para identificar el estudio (cambia este nº en cada estudio)

# Horizontes de predicción en nº de pasos (1=10 min, 6=1h, 12=2h, 24=4h, 48=8h)
PREDICTION_HORIZONS = [1, 6, 12, 24, 48]
#PREDICTION_HORIZONS = [1]  # prueba rápida 
# Hipers del Transformer (pon aquí tus valores definitivos)
ARCHITECTURE = "encoder"  # "encoder" o "encdec"
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT_RATE = 0.0

# Entrenamiento (ajústalos a tu gusto)
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VERBOSE_FIT = 0  # 0: silencioso, 1: barras de progreso

# =========================
#   PREPROCESADO
# =========================
from scripts.preprocess_data_Tetuan import preprocess_data_electricity

df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)

# =========================
#   SECUENCIAS MULTI-STEP
# =========================
def create_multistep_sequences(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        # y + exógenas (Usage_kWh en col 0, exógenas desde col 3)
        seq = np.concatenate([data[i:i + seq_length + pred_length, 0:1],data[i:i + seq_length + pred_length, 3:]],axis=1).copy()
        seq[-pred_length:, 0] = 0 # enmascara el target futuro
        X.append(seq)
        y.append(data[i + seq_length:i + seq_length + pred_length, 0])
    return np.array(X), np.array(y)

# =========================
#   BLOQUES TRANSFORMER
# =========================
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_encoder_block(inputs, num_heads, ff_dim, dropout_rate):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# =========================
#   MODELOS
# =========================
def build_encoder_model(seq_length, num_features, d_model, num_heads, ff_dim, num_layers, dropout_rate, pred_length):
    inputs = Input(shape=(seq_length, num_features))
    x = Dense(d_model)(inputs)
    x = x + positional_encoding(seq_length, d_model)
    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(ff_dim, activation='relu')(x)
    outputs = Dense(pred_length)(x)  # multistep de una sola tirada
    return Model(inputs, outputs)

def build_encdec_model(seq_length, num_features, d_model, num_heads, ff_dim, num_layers, dropout_rate, pred_length):
    # Encoder
    enc_inputs = Input(shape=(seq_length, num_features))
    x = Dense(d_model)(enc_inputs)
    x = x + positional_encoding(seq_length, d_model)
    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)
    context = GlobalAveragePooling1D()(x)

    # Decoder "simple": repetir contexto y proyectar a la secuencia de salida
    dec = RepeatVector(pred_length)(context)  # (batch, pred_length, d_model? no, ahora es 1D -> lo expandimos con Dense)
    dec = TimeDistributed(Dense(ff_dim, activation='relu'))(dec)
    dec = TimeDistributed(Dense(1))(dec)     # salida (batch, pred_length, 1)
    outputs = Flatten()(dec)                 # (batch, pred_length)
    return Model(enc_inputs, outputs)

def build_transformer(seq_in, num_features, pred_length):
    if ARCHITECTURE == "encoder":
        return build_encoder_model(seq_in, num_features, D_MODEL, NUM_HEADS, FF_DIM, NUM_LAYERS, DROPOUT_RATE, pred_length)
    elif ARCHITECTURE == "encdec":
        return build_encdec_model(seq_in, num_features, D_MODEL, NUM_HEADS, FF_DIM, NUM_LAYERS, DROPOUT_RATE, pred_length)
    else:
        raise ValueError("ARCHITECTURE debe ser 'encoder' o 'encdec'")

# =========================
#   ESTUDIO AHEAD
# =========================
results = []
for pred_len in PREDICTION_HORIZONS:
    print(f"[Transformer] Entrenando para predicción a {pred_len} pasos...")
    X_seq, y_seq = create_multistep_sequences(X_exog, SEQ_LENGTH, pred_len)

    # Para el modelo usamos solo la parte visible por el encoder (SEQ_LENGTH)
    # El create_multistep_sequences creó ventanas de (SEQ_LENGTH + pred_len);
    # mantendremos SEQ_LENGTH como entrada (asumiendo que exógenas futuras se quieran usar,
    # podrías cambiar a usar toda la ventana). Por defecto aquí usamos TODA la ventana
    # para permitir exógenas futuras (como en el LSTM_Ahead).

    X_in = X_seq  # (batch, SEQ_LENGTH + pred_len, features)
    seq_in_len = X_in.shape[1]  # = SEQ_LENGTH + pred_len

    X_train, X_test, y_train, y_test = train_test_split(X_in, y_seq, test_size=0.2, random_state=42)
    num_features = X_in.shape[2]

    model = build_transformer(seq_in_len, num_features, pred_len)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])

    start_time = time()
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test), verbose=VERBOSE_FIT)
    training_time = time() - start_time

    preds = model.predict(X_test, verbose=0)

    # Desnormalizar (aplanamos para comparar globalmente todas las horizontes)
    y_test_flat = y_test.flatten().reshape(-1, 1)
    pred_flat = preds.flatten().reshape(-1, 1)

    y_test_real = scaler_y.inverse_transform(
        np.hstack((y_test_flat, np.zeros((len(y_test_flat), 2))))
    )[:, 0]
    preds_real = scaler_y.inverse_transform(
        np.hstack((pred_flat, np.zeros((len(pred_flat), 2))))
    )[:, 0]

    mse = mean_squared_error(y_test_real, preds_real)
    r2 = r2_score(y_test_real, preds_real)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_real - preds_real) / y_test_real)) * 100
    mae = np.mean(np.abs(y_test_real - preds_real))
    cvrmse = np.std((y_test_real - preds_real) / y_test_real) * 100

    train_loss_final = history.history['loss'][-1]
    val_loss_final = history.history['val_loss'][-1]
    overfit_gap = val_loss_final - train_loss_final
    loss_ratio = val_loss_final / train_loss_final if train_loss_final != 0 else np.nan

    results.append({
        'architecture': ARCHITECTURE,
        'd_model': D_MODEL,
        'num_heads': NUM_HEADS,
        'ff_dim': FF_DIM,
        'num_layers': NUM_LAYERS,
        'dropout_rate': DROPOUT_RATE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'prediction_horizon': pred_len,
        'MSE': mse,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'CVRMSE': cvrmse,
        'Training Time (s)': training_time,
        'Train Loss Final': train_loss_final,
        'Validation Loss Final': val_loss_final,
        'Overfit Gap': overfit_gap,
        'Loss Ratio': loss_ratio,
    })

    #model.save(f"models/Transformer_ahead_{NUM_CONFIGURACION_ESTUDIO}_{pred_len}.h5")
    model.save(f"models/Transformer_ahead_{NUM_CONFIGURACION_ESTUDIO}_{pred_len}.keras")  # Keras v3



# Guardar resultados
results_df = pd.DataFrame(results)
results_df.to_csv(f"./output/AheadDL/Transformer_multistep_prediction_results_{NUM_CONFIGURACION_ESTUDIO}.csv", index=False)

print("Estudio de predicción multi-horizonte con Transformers completado.")
