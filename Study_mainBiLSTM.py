import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dropout, Dense, Input, Bidirectional, Multiply, Flatten)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from time import time

from scripts.preprocess_data_Tetuan import preprocess_data_electricity


#tf.debugging.set_log_device_placement(True)
# Cargar datos
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')

# Preprocesar datos
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)

# Crear secuencias
def create_sequences_old(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(np.concatenate([data[i:i + seq_length, 0:1], data[i:i + seq_length, 3:]], axis=1))
        labels.append(data[i + seq_length, 0])
    return np.array(sequences), np.array(labels)

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

SEQ_LENGTH = 6*8
X_seq, y_seq = create_sequences(X_exog, SEQ_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

num_features = X_exog.shape[1]-2

# Atención característica
def feature_attention(inputs):
    attention = Dense(inputs.shape[-1], activation='sigmoid')(inputs)
    attention = Dense(inputs.shape[-1], activation='softmax')(attention)
    return Multiply()([inputs, attention])

# Atención temporal
def temporal_attention(inputs):
    seq_length = inputs.shape[1]
    attention = Dense(1, activation='relu')(inputs)
    attention = Flatten()(attention)
    attention = Dense(seq_length, activation='softmax')(attention)
    attention = tf.keras.layers.Reshape((seq_length, 1))(attention)
    return Multiply()([inputs, attention])

# Crear modelo BiLSTM con atención (varía unidades para estudio)
def build_bilstm_model(seq_length, num_features, units_lstm1, units_lstm2, dropout_rate, use_feature_att, use_temporal_att):
    inputs = Input(shape=(seq_length, num_features))
    bi_lstm = Bidirectional(LSTM(units_lstm1, return_sequences=True))(inputs)
    bi_lstm = Dropout(dropout_rate)(bi_lstm)

    if use_feature_att:
        bi_lstm = feature_attention(bi_lstm)

    bi_lstm2 = Bidirectional(LSTM(units_lstm2, return_sequences=True))(bi_lstm)

    if use_temporal_att:
        bi_lstm2 = temporal_attention(bi_lstm2)

    flattened = Flatten()(bi_lstm2)
    dense1 = Dense(1024, activation='relu')(flattened)
    output = Dense(1)(dense1)

    return Model(inputs, output)

# Parámetros de estudio
units_lstm_values = [(64,32), (128,64), (256,128)]
dropout_values = [0, 0.2]
#learning_rate = 0.001
epochs_values = [10, 20, 50]
batch_sizes = [32, 64]
attention_options = [(True, True), (True, False), (False, True),(False, False)]

results = []
ii=1
# Estudio de ablación
for units_lstm1, units_lstm2 in units_lstm_values:
    for dropout_rate in dropout_values:
        for epochs in epochs_values:
            for batch_size in batch_sizes:
                for use_feature_att, use_temporal_att in attention_options:
                    #ii += 1
                    #if ii < 119:  # Continuar si el índice es menor a 76
                    #    continue
                    model = build_bilstm_model(
                        SEQ_LENGTH, num_features,
                        units_lstm1, units_lstm2,
                        dropout_rate,
                        use_feature_att, use_temporal_att
                    )
                    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

                    start_time = time()
                    history = model.fit(X_train, y_train, epochs = epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
                    # Crear datasets optimizados para input pipeline
                    #train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
                    #    .shuffle(buffer_size=1000) \
                    #    .batch(batch_size) \
                    #    .prefetch(tf.data.AUTOTUNE)
#
                    #test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
                    #    .batch(batch_size) \
                    #    .prefetch(tf.data.AUTOTUNE)
#
                    ## Entrenamiento con datasets optimizados
                    #history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=0)
                    
                    training_time = time() - start_time

                    predictions = model.predict(X_test)
                    y_test_real = scaler_y.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 2)))))[:, 0]
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
                        'units_lstm1': units_lstm1,
                        'units_lstm2': units_lstm2,
                        'dropout_rate': dropout_rate,
                        'feature_att': use_feature_att,
                        'temporal_att': use_temporal_att,
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
                        'Loss Ratio': loss_ratio
                    })
                    pd.DataFrame(results).to_csv('./output/ResultsDL/01_BiLSTM_ablation_results_new2.csv', index=False)
                    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")

print("Estudio de ablación BiLSTM completado y guardado.")
