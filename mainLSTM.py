import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from time import time
import joblib
import matplotlib.pyplot as plt

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

# Definir el modelo LSTM
dropout = False  # Añadir Dropout
num_features = X_exog.shape[1]  # Número de características en X
layers = [
            LSTM(128, return_sequences=False, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            Dense(8, activation='relu'),
            Dense(1)
        ]
model = Sequential([layer for layer in layers if layer is not None])

# Compilar el modelo
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

model.summary()  # Mostrar resumen del modelo
epocas = 50  # Número de épocas para el entrenamiento
tamañoBatch = 64


start_time = time()  # Registrar el tiempo de inicio
# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=epocas, batch_size=tamañoBatch, validation_data=(X_test, y_test))
end_time = time()  # Registrar el tiempo de fin
training_time = end_time - start_time  # Calcular el tiempo total de entrenamiento

model.save('models/modelo_LSTM_numlayers1.h5')
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')

# Hacer predicciones
predictions = model.predict(X_test)

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
if 1:
    print(f'MSE: {mse:.2f}') # Error cuadrático medio
    print(f'R2: {r2:.2f}') # Coeficiente de determinación
    print(f'RMSE: {rmse:.2f}')  # Raíz del error cuadrático medio
    print(f'MAPE: {mape:.2f}')  # Error porcentual absoluto medio
    print(f'MAE: {mae:.2f}')  # Error absoluto medio
    print(f'MRE: {mre:.2f}')  # Error relativo medio
    print(f'CVRMSE: {cvrmse:.2f}')  # Coeficiente de variación del error cuadrático medio
    print(f'Tiempo de entrenamiento: {training_time:.2f} segundos') # Tiempo de entrenamiento
    #otra prueba

    # Graficar la función de pérdida (MSE) para entrenamiento y validación
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.title('Curva de pérdida entrenamiento vs validación')
    plt.legend()
    plt.grid()
    plt.show()
    # Guardar los resultados junto con la configuración y el tiempo de entrenamiento
    # Lista para guardar los resultados
    if 0:
        results = []
        results.append({
            'num_layers': 2,
            'dropout': dropout,
            'epochs': epocas,
            'batch_size': tamañoBatch,
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
        #results_df.to_csv('./output/ResultsDL/ResultTetuan20.csv', index=False)
# Guardar el modelo en formato HDF5
