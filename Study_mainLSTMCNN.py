import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error
from time import time # Importa time para medir el tiempo de entrenamiento

# Importar la función de preprocesamiento
from scripts.preprocess_data_steel import preprocess_data_steel
from scripts.preprocess_data_Tetuan import preprocess_data_electricity
from scripts.generacionModelosLSTMCNN import seleccionModelosHibridos # Importar la función que genera los modelos LSTM para el estudio de ablación

# Cargar datos
#df = pd.read_csv('data/00_Steel_industry_data.csv')
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')

# Para el modelo LSTM queremos incluir la propia variable target (Usage_kWh)
# dentro de la secuencia, así la usamos como referencia en la secuencia.
# Insertamos 'Usage_kWh' como primera columna en X_exog.
# X_exog.insert(0, 'Usage_kWh', y)

# Preprocesar datos
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)

# Función para crear secuencias de datos
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(np.concatenate([data[i:i + seq_length, 0:1], data[i:i + seq_length, 3:]], axis=1))
        # Suponemos que el valor a predecir es el primer valor de la siguiente fila (Usage_kWh normalizado)
        labels.append(data[i + seq_length, 0])
    return np.array(sequences), np.array(labels)

# Crear secuencias: por ejemplo, secuencia de 24 pasos
SEQ_LENGTH = 6*8  # cada 10 mins los datos
X_seq, y_seq = create_sequences(X_exog, SEQ_LENGTH)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

epochs_values = [10,20,50]
epochs_values = [50]
batch_size_values = [64]

# Lista para guardar los resultados
results = []

#- Vamos al lio
num_features = X_exog.shape[1]-2

# Bucle principal para iterar sobre las configuraciones del modelo
for config in [4]:  # Las 5 configuraciones definidas
    for dropout in [ False]:  # Con y sin Dropout
        model = seleccionModelosHibridos(
            flagModelo=config,
            num_features=num_features,
            SEQ_LENGTH=SEQ_LENGTH,
            dropout=dropout
        )
        # Compilar el modelo
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.summary()  # Mostrar resumen del modelo
        for epochs in epochs_values :
            for batch_size in batch_size_values:
                # Compilar el modelo
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
                #y_test_real = scaler_y.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), X_exog.shape[1]-1)))))[:, 0]
                #predictions_real = scaler_y.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), X_exog.shape[1]-1)))))[:, 0]
                # Invertir la normalización para obtener valores reales
                y_test_real = scaler_y.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 2)))))[:, 0]
                predictions_real = scaler_y.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), 2)))))[:, 0]
                # Calcular MSE, R2, MAE y RMSE
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

                # Convertir los resultados en un DataFrame para analizarlos fácilmente
                results_df = pd.DataFrame(results)

                # Guardar los resultados en un archivo CSV
                results_df.to_csv('./output/ResultsDL/01_LSTMCNN_ablation_result_4.csv', index=False)

print("Resultados guardados en 'Tetuan_model_ablation_results_with_time.csv'")

