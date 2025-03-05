import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error
from time import time # Importa time para medir el tiempo de entrenamiento

# Importar la función de preprocesamiento
from scripts.preprocess_data_steel import preprocess_data_steel
from scripts.generacionModelosLSTM import seleccionModelos # Importar la función que genera los modelos LSTM para el estudio de ablación

# Cargar datos
df = pd.read_csv('data/00_Steel_industry_data.csv')

# Preprocesar datos usando preprocess_data_steel
# X_exog contendrá las features adicionales e y es 'Usage_kWh'
X_exog, y = preprocess_data_steel(df)

# Para el modelo LSTM queremos incluir la propia variable target (Usage_kWh)
# dentro de la secuencia, así la usamos como referencia en la secuencia.
# Insertamos 'Usage_kWh' como primera columna en X_exog.
X_exog.insert(0, 'Usage_kWh', y)

# Convertir a array y normalizar todos los features (incluyendo Usage_kWh)
#scaler = MinMaxScaler()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X_exog)

# Función para crear secuencias de datos
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        # Suponemos que el valor a predecir es el primer valor de la siguiente fila (Usage_kWh normalizado)
        labels.append(data[i + seq_length, 0])
    return np.array(sequences), np.array(labels)

# Crear secuencias de 8h (4*8=32 pasos)
NumHoras = 8
SEQ_LENGTH = 4*NumHoras
X_seq, y_seq = create_sequences(data_scaled, SEQ_LENGTH)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

epochs_values = [10,20, 50]
batch_size_values = [16, 32, 64]

# Lista para guardar los resultados
results = []

# Vamos al lio

# Bucle principal para iterar sobre las configuraciones del modelo
for num_layers in range(1, 6):  # 1 a 5 capas LSTM
    for dropout in [True, False]:  # Con y sin Dropout
        for epochs in epochs_values:
            for batch_size in batch_size_values:
                # Definir el modelo LSTM según la configuración
                model = seleccionModelos(num_layers, X_exog.shape[1], SEQ_LENGTH, dropout)

                # Compilar el modelo
                model.compile(optimizer='adam', loss='mse')

                # Medir el tiempo de entrenamiento
                start_time = time()  # Registrar el tiempo de inicio

                # Entrenar el modelo
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

                end_time = time()  # Registrar el tiempo de fin
                training_time = end_time - start_time  # Calcular el tiempo total de entrenamiento

                # Evaluación en el conjunto de prueba
                #loss, mae = model.evaluate(X_test, y_test)
                
                # Hacer predicciones
                predictions = model.predict(X_test)

                # Invertir la normalización para obtener valores reales
                y_test_real = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), X_exog.shape[1]-1)))))[:, 0]
                predictions_real = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), X_exog.shape[1]-1)))))[:, 0]

                # Calcular MSE, R2, MAE y RMSE
                mse = mean_squared_error(y_test_real, predictions_real)
                r2 = r2_score(y_test_real, predictions_real)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
                mae = np.mean(np.abs(y_test_real - predictions_real))
                # Guardar los resultados junto con la configuración y el tiempo de entrenamiento
                results.append({
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'MSE': mse,
                    'R2': r2,
                    'MAE': mae,
                    'MAPE': mape,
                    #'loss': loss,
                    'RMSE': rmse,
                    'Training Time (s)': training_time
                })

# Convertir los resultados en un DataFrame para analizarlos fácilmente
results_df = pd.DataFrame(results)

# Guardar los resultados en un archivo CSV
results_df.to_csv('./output/ResultsDL/model_ablation_results_with_time.csv', index=False)

print("Resultados guardados en 'model_ablation_results_with_time.csv'")

