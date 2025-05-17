import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from time import time
from scripts.preprocess_data_Tetuan import preprocess_data_electricity

# Cargar datos y preprocesarlos
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)

# # Aplicar diferenciación de primer orden al conjunto de features
# def difference(data):
#     # Calcula la diferencia a lo largo del tiempo
#     diff = np.diff(data, axis=0)
#     # Prepara la primera fila (se puede rellenar con ceros)
#     diff = np.vstack([np.zeros((1, data.shape[1])), diff])
#     return diff

# X_exog_diff = difference(X_exog)

# Función para crear secuencias utilizando ventana deslizante
# Función para crear secuencias de datos
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(np.concatenate([data[i:i + seq_length, 0:1], data[i:i + seq_length, 3:]], axis=1))
        # Suponemos que el valor a predecir es el primer valor de la siguiente fila (Usage_kWh normalizado)
        labels.append(data[i + seq_length, 0])
    return np.array(sequences), np.array(labels)

# def create_sequences_sliding(data, seq_length, output_length):
#     sequences = []
#     labels = []
#     for i in range(len(data) - seq_length - output_length + 1):  
#         # Ventana de entrada (features desde la columna 2 en adelante)
#         window = data[i:i+seq_length]  # Excluyendo las primeras 3 columnas
#         sequences.append(window)
#         # Ventana de salida (futuros `output_length` valores de la columna 2)
#         labels.append(data[i+seq_length:i+seq_length+output_length, 0])  
#     return np.array(sequences), np.array(labels)

numHoras= 8
SEQ_LENGTH = 6 * numHoras  # cada 10 mins los datos

X_seq, y_seq = create_sequences(X_exog, SEQ_LENGTH)


X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=41)

# Definir un modelo CNN-LSTM basado en el paper:
# Se comienza con capas convolucionales que extraen características locales, luego se procesa la secuencia con una LSTM.
num_features = X_exog.shape[1]-2
model = Sequential([
    # Capa CNN
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(SEQ_LENGTH, num_features)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    # Capa LSTM
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    # Capas densas (como en el paper)
    Dense(2048, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(1)
])

# Definir el optimizador Adam con un learning rate específico
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

start_time = time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
training_time = time() - start_time

predictions = model.predict(X_test)

# Adaptar la inversión de la normalización para secuencias (OUTPUT_LENGTH valores por muestra)
# Para predicciones:
predictions_flat = predictions.reshape(-1, 1)  # Forma: (N*OUTPUT_LENGTH, 1)
zeros_pred = np.zeros((predictions_flat.shape[0], 2))  # Mismo número de filas
predictions_extended = np.hstack((predictions_flat, zeros_pred))  # (N*OUTPUT_LENGTH, 3)
predictions_inverted = scaler_y.inverse_transform(predictions_extended)[:, 0]
predictions_real = predictions_inverted.reshape(predictions.shape)  # Volver a (N, OUTPUT_LENGTH)

# Para y_test:
y_test_flat = y_test.reshape(-1, 1)
zeros_y = np.zeros((y_test_flat.shape[0], 2))
y_test_extended = np.hstack((y_test_flat, zeros_y))
y_test_inverted = scaler_y.inverse_transform(y_test_extended)[:, 0]
y_test_real = y_test_inverted.reshape(y_test.shape)

# Calcular MSE, R2, MAE y RMSE
mse = mean_squared_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
mae = np.mean(np.abs(y_test_real - predictions_real))
mre = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
cvrmse = np.std((y_test_real - predictions_real) / y_test_real) * 100
sde = np.std(y_test_real - predictions_real)

if 1:
    print(f'MSE: {mse:.2f}') # Error cuadrático medio
    print(f'R2: {r2:.2f}') # Coeficiente de determinación
    print(f'RMSE: {rmse:.2f}')  # Raíz del error cuadrático medio
    print(f'MAPE: {mape:.2f}')  # Error porcentual absoluto medio
    print(f'MAE: {mae:.2f}')  # Error absoluto medio
    print(f'MRE: {mre:.2f}')  # Error relativo medio
    print(f'CVRMSE: {cvrmse:.2f}')  # Coeficiente de variación del error cuadrático medio
    print(f'SDE: {sde:.2f}')  # Coeficiente de variación del error cuadrático medio
    print(f'Tiempo de entrenamiento: {training_time:.2f} segundos') # Tiempo de entrenamiento
    
    # Guardar los resultados junto con la configuración y el tiempo de entrenamiento
    # Lista para guardar los resultados
else:
    results = []
    results.append({
        'num_layers': 'LSTM+CNN',
        'dropout': dropout_rate,
        'epochs': epoch_rate,
        'batch_size': batch_size_rate,
        'MSE': mse,
        'R2': r2,
        'MAE': mae,
        'MAPE': mape,
        'MRE': mre,
        #'loss': loss,
        'RMSE': rmse,
        'CVRMSE': cvrmse,
        'SDE': sde,
        'Training Time (s)': training_time
        })  
    # Convertir los resultados en un DataFrame para analizarlos fácilmente
    results_df = pd.DataFrame(results)
    # Guardar los resultados en un archivo CSV
    results_df.to_csv('./output/ResultsDL/01_CNNLSTM_prueba.csv', index=False)