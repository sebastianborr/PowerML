import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add, RepeatVector, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scripts.preprocess_data_Tetuan import preprocess_data_electricity
from time import time
import os

# ------------------ Datos ------------------
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)

# Crear secuencias
def create_sequences_old(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = np.concatenate([data[i:i+seq_length, 0:1], data[i:i+seq_length, 3:]], axis=1)
        X.append(seq)
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        #sequences.append(data[i:i + seq_length])
        seq = np.concatenate([data[i:i + seq_length, 0:1], data[i:i + seq_length, 3:]], axis=1).copy()
        seq[-1, 0] = 0  # <--- Pone a cero la variable objetivo en el último paso
        sequences.append(seq)
        # Suponemos que el valor a predecir es el primer valor de la siguiente fila (Usage_kWh normalizado)
        labels.append(data[i + seq_length, 0])
    return np.array(sequences), np.array(labels)

SEQ_LENGTH = 48
X_seq, y_seq = create_sequences(X_exog, SEQ_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)


# ------------------ Positional Encoding ------------------
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# ------------------ Encoder Block ------------------
def transformer_encoder_block(inputs, num_heads, ff_dim, dropout_rate):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# ------------------ Encoder Model ------------------
def build_encoder_model(seq_length, num_features, d_model, num_heads, ff_dim, num_layers, dropout_rate):
    inputs = Input(shape=(seq_length, num_features))
    x = Dense(d_model)(inputs)
    x += positional_encoding(seq_length, d_model)
    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(ff_dim, activation='relu')(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

# ------------------ Encoder-Decoder Model ------------------
def build_encdec_model(seq_length, num_features, d_model, num_heads, ff_dim, num_layers, dropout_rate):
    encoder_inputs = Input(shape=(seq_length, num_features))
    x = Dense(d_model)(encoder_inputs)
    x += positional_encoding(seq_length, d_model)
    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)

    context_vector = GlobalAveragePooling1D()(x)
    decoder_inputs = RepeatVector(1)(context_vector)
    decoder_outputs = TimeDistributed(Dense(ff_dim, activation='relu'))(decoder_inputs)
    decoder_outputs = TimeDistributed(Dense(1))(decoder_outputs)
    outputs = tf.keras.layers.Flatten()(decoder_outputs)
    return Model(encoder_inputs, outputs)


# ------------------ Configuraciones de Ablación ------------------
architectures = ['encoder', 'encdec']
d_models = [32, 64]
num_heads_list = [2, 4]
ff_dims = [64, 128]
num_layers_list = [1, 2]
dropout_rates = [0, 0.2]
batch_sizes = [64,128]
epochs_t = [20,50]
results = []
# os.makedirs('./output/ResultsDL', exist_ok=True)
ii = 0
for arch in architectures:
    for d_model in d_models:
        for num_heads in num_heads_list:
            for ff_dim in ff_dims:
                for num_layers in num_layers_list:
                    for dropout_rate in dropout_rates:
                        for epochs in epochs_t:
                            for batch_size in batch_sizes:
                                ii += 1 
                                #if ii<543:
                                #    continue
                                print(f"Estudio de ablación {ii} /576")
                                if arch == 'encoder':
                                    model = build_encoder_model(SEQ_LENGTH, X_seq.shape[2], d_model, num_heads, ff_dim, num_layers, dropout_rate)
                                else:
                                    model = build_encdec_model(SEQ_LENGTH, X_seq.shape[2], d_model, num_heads, ff_dim, num_layers, dropout_rate)
                                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

                                start_time = time()
                                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
                                training_time = time() - start_time

                                predictions = model.predict(X_test)
                                y_test_real = scaler_y.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros((len(y_test), 2)))))[:, 0]
                                predictions_real = scaler_y.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), 2)))))[:, 0]

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
                                loss_ratio = val_loss_final / train_loss_final if train_loss_final != 0 else np.nan

                                # Imprimir si hay indicios de overfitting
                                if loss_ratio > 1.2:  # por ejemplo, si la loss de validación es 20% mayor que la de entrenamiento
                                    print(f"Posible overfitting: Loss Ratio = {loss_ratio:.2f} (Gap = {overfit_gap:.2f})")
                                else:
                                    print(f"No hay overfitting. Loss Ratio = {loss_ratio:.2f} (Gap = {overfit_gap:.2f})")
                                results.append({
                                    'architecture': arch,
                                    'd_model': d_model,
                                    'num_heads': num_heads,
                                    'ff_dim': ff_dim,
                                    'num_layers': num_layers,
                                    'dropout_rate': dropout_rate,
                                    'epochs': epochs,
                                    'batch_size': batch_size,
                                    'MSE': mse,
                                    'R2': r2,
                                    'MAE': mae,
                                    'MAPE': mape,
                                    'MRE': mre,
                                    #'loss': loss,
                                    'RMSE': rmse,
                                    'CVRMSE': cvrmse,
                                    'Training Time (s)': training_time,
                                    'Train Loss Final': train_loss_final,
                                    'Validation Loss Final': val_loss_final,
                                    'Overfit Gap': overfit_gap,
                                    'Loss Ratio': loss_ratio,
                                })
                                pd.DataFrame(results).to_csv('./output/ResultsDL/01_transformer_ablation_results_new.csv', index=False)

print("Estudio de ablación de Transformers completado.")
