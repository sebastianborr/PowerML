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

# Cargar datos
df = pd.read_csv('data/00_Steel_industry_data.csv')

# Preprocesar datos usando preprocess_data_steel
# X_exog contendrá las features adicionales y y es 'Usage_kWh'
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

# Crear secuencias: por ejemplo, secuencia de 24 pasos
SEQ_LENGTH = 32
X_seq, y_seq = create_sequences(data_scaled, SEQ_LENGTH)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Definir el modelo LSTM
dropout = True
layers = [
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, X_exog.shape[1])),
    Dropout(0.2) if dropout else None,
    LSTM(64, return_sequences=False),
    Dropout(0.2) if dropout else None,
    Dense(8, activation='relu'),
    Dense(1)
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
y_test_real = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), X_exog.shape[1]-1)))))[:, 0]
predictions_real = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), X_exog.shape[1]-1)))))[:, 0]

# Calcular MSE, R2, MAE y RMSE
mse = mean_squared_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
mae = np.mean(np.abs(y_test_real - predictions_real))

if 1:
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Training Time (s): {training_time:.4f}")

