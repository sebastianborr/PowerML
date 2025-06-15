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
from scripts.generacionModelosLSTM import seleccionModelos # Importar la función que genera los modelos LSTM para el estudio de ablación
from scripts.preprocess_data_Tetuan import preprocess_data_electricity

# Cargar datos
#df = pd.read_csv('data/00_Steel_industry_data.csv')
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')

# Para el modelo LSTM queremos incluir la propia variable target (Usage_kWh)
# dentro de la secuencia, así la usamos como referencia en la secuencia.
# Insertamos 'Usage_kWh' como primera columna en X_exog.
# X_exog.insert(0, 'Usage_kWh', y)

# Preprocesar datos
X_exog, y, scaler_X, scaler_y = preprocess_data_electricity(df)

# Función para crear secuencias de datos de entrada y salida del dato a predecir 0
def create_sequence_old(data, seq_length):
    sequences = [] 
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(np.concatenate([data[i:i + seq_length, 0:1], data[i:i + seq_length, 3:]], axis=1))
        # Suponemos que el valor a predecir es el primer valor de la siguiente fila (Usage_kWh normalizado)
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

# Crear secuencias: por ejemplo, secuencia de 24 pasos
SEQ_LENGTH = 6*8
X_seq, y_seq = create_sequences(X_exog, SEQ_LENGTH)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

epochs_values = [10, 20, 50]
batch_size_values = [16, 32, 64]
dropout_values = [True, False]
num_layers_values = [1,2,3,4,5]
# Lista para guardar los resultados
results = []
ii=1

# Vamos al lio
num_features = X_exog.shape[1]-2
# Bucle principal para iterar sobre las configuraciones del modelo
for num_layers in num_layers_values: #range(1, 4):  # 1 a 5 capas LSTM
    for dropout in dropout_values:  # Con y sin Dropout
        for epochs in epochs_values:
            for batch_size in batch_size_values:
                if ii<76:
                    ii+=1
                    continue
                ii+=1
                print(f"Configuración {ii}")
                # Definir el modelo LSTM según la configuración
                model = seleccionModelos(num_layers,num_features , SEQ_LENGTH, dropout)

                # Compilar el modelo
                optimizer = Adam(learning_rate=0.001)
                model.compile(optimizer=optimizer, loss='mse')

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

                train_loss_final = history.history['loss'][-1]
                val_loss_final = history.history['val_loss'][-1]
                overfit_gap = val_loss_final - train_loss_final
                loss_ratio = val_loss_final / train_loss_final if train_loss_final != 0 else np.nan

                # Imprimir si hay indicios de overfitting
                if loss_ratio > 1.2:  # por ejemplo, si la loss de validación es 20% mayor que la de entrenamiento
                    print(f"Posible overfitting: Loss Ratio = {loss_ratio:.2f} (Gap = {overfit_gap:.2f})")
                else:
                    print(f"No hay overfitting. Loss Ratio = {loss_ratio:.2f} (Gap = {overfit_gap:.2f})")
    
                
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

                # Convertir los resultados en un DataFrame para analizarlos fácilmente
                results_df = pd.DataFrame(results)

                # Guardar los resultados en un archivo CSV
                results_df.to_csv('./output/ResultsDL/01_LSTM_MultiLayer_ablation_new3.csv', index=False)

print("Resultados guardados en 'Tetuan_model_ablation_results_with_time.csv'")

