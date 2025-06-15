import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from time import time
from scripts.preprocess_data_Tetuan import preprocess_data_electricity
from scripts.generacionModelosCNNLSTM import seleccionModelosCNN_LSTM

# Cargar datos
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')

# Preprocesar datos
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)

# Función para crear secuencias
def create_sequences_old(data, seq_length):
    sequences = []
    labels = []
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
# Parámetros
numHoras = 8
SEQ_LENGTH = 6 * numHoras  # 8 horas de datos (48 pasos de 10 minutos)
X_seq, y_seq = create_sequences(X_exog, SEQ_LENGTH)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=41)

# Configuraciones para el estudio de ablación
epochs_values = [10, 20, 50]
batch_size_values = [16, 32, 64]
dropout_values = [True, False]

# Bucle principal para iterar sobre las configuraciones
results = []
num_features = X_exog.shape[1]-2

for config in [1, 2, 3, 4]:  # Las 4 configuraciones definidas
    for dropout in dropout_values:
        for epochs in epochs_values:
            for batch_size in batch_size_values:
                model = seleccionModelosCNN_LSTM(
                    flagModelo=config,
                    num_features=num_features,
                    SEQ_LENGTH=SEQ_LENGTH,
                    dropout=dropout
                )
                
                # Compilar el modelo
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                model.summary()
                
                # Entrenar y medir tiempo
                start_time = time()
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                  validation_data=(X_test, y_test), verbose=1)
                training_time = time() - start_time
                
                # Evaluar el modelo
                predictions = model.predict(X_test)
                
                # Invertir la normalización
                y_test_real = scaler_y.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 2)))))[:, 0]
                predictions_real = scaler_y.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), 2)))))[:, 0]
                
                mse = mean_squared_error(y_test_real, predictions_real)
                r2 = r2_score(y_test_real, predictions_real)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
                mae = np.mean(np.abs(y_test_real - predictions_real))
                mre = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
                cvrmse = np.std((y_test_real - predictions_real) / y_test_real) * 100
                # Guardar los resultados junto con la configuración y el tiempo de entrenamiento
                results.append({
                    'num_layers': config,
                    'dropout': dropout,
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
                    'Training Time (s)': training_time
                })
                # Guardar resultados en CSV
                results_df = pd.DataFrame(results)
                results_df.to_csv('./output/ResultsDL/01_CNNLSTM_ablation_results_new.csv', index=False)

print("Resultados guardados en 'CNNLSTM_ablation_results.csv'")