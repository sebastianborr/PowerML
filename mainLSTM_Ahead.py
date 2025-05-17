import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from time import time
# Importar la función de preprocesamiento
from scripts.preprocess_data_steel import preprocess_data_steel
from scripts.preprocess_data_Tetuan import preprocess_data_electricity

# Cargar datos
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')

# Para el modelo LSTM queremos incluir la propia variable target (Usage_kWh)
# dentro de la secuencia, así la usamos como referencia en la secuencia.
# Insertamos 'Usage_kWh' como primera columna en X_exog.
# Preprocesar datos
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)
# Función para crear secuencias de datos
def create_sequences(data, seq_length, pred_length=6):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - pred_length + 1):
        sequences.append(data[i:i + seq_length])
        # Ahora la etiqueta es una secuencia de pred_length valores
        labels.append(data[i + seq_length:i + seq_length + pred_length, 0])
    return np.array(sequences), np.array(labels)

# Crear secuencias: por ejemplo, secuencia de 24 pasos
SEQ_LENGTH = 48 # cada 10 mins los datos
PRED_LENGTH = 12 # cada 10 mins los datos
X_seq, y_seq = create_sequences(X_exog, SEQ_LENGTH,PRED_LENGTH)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Definir el modelo LSTM
dropout = True  # Añadir Dropout
num_features = X_exog.shape[1]  # Número de características en X
layers = [
            LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            LSTM(64, return_sequences=False),
            Dropout(0.2) if dropout else None,
            Dense(32, activation='relu'),
            Dense(PRED_LENGTH) #capa de salida con PRED_LENGTH neuronas
        ]
model = Sequential([layer for layer in layers if layer is not None])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


start_time = time()  # Registrar el tiempo de inicio
# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
end_time = time()  # Registrar el tiempo de fin
training_time = end_time - start_time  # Calcular el tiempo total de entrenamiento

# Hacer predicciones
predictions = model.predict(X_test)

# Invertir la normalización para obtener valores reales
y_test_real = scaler_y.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test)*PRED_LENGTH, 2)))))[:, 0]
predictions_real = scaler_y.inverse_transform(np.hstack((predictions.reshape(-1, 1), np.zeros((len(predictions)*PRED_LENGTH, 2)))))[:, 0]

# Calcular MSE, R2, MAE y RMSE
mse = mean_squared_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
mae = np.mean(np.abs(y_test_real - predictions_real))
mre = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
cvrmse = np.std((y_test_real - predictions_real) / y_test_real) * 100
if 1:
    print(f'MSE: {mse:.2f}') # Error cuadrático medio
    print(f'R2: {r2:.2f}') # Coeficiente de determinación
    print(f'RMSE: {rmse:.2f}')  # Raíz del error cuadrático medio
    print(f'MAPE: {mape:.2f}')  # Error porcentual absoluto medio
    print(f'MAE: {mae:.2f}')  # Error absoluto medio
    print(f'MRE: {mre:.2f}')  # Error relativo medio
    print(f'CVRMSE: {cvrmse:.2f}')  # Coeficiente de variación del error cuadrático medio
    print(f'Tiempo de entrenamiento: {training_time:.2f} segundos') # Tiempo de entrenamiento
    
    # Guardar los resultados junto con la configuración y el tiempo de entrenamiento
    # Lista para guardar los resultados
    if 0:
        results = []
        results.append({
            'num_layers': 2,
            'dropout': dropout,
            'epochs': 20,
            'batch_size': 32,
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
        # Convertir los resultados en un DataFrame para analizarlos fácilmente
        results_df = pd.DataFrame(results)
        # Guardar los resultados en un archivo CSV
        results_df.to_csv('./output/ResultsDL/ResultTetuan20.csv', index=False)