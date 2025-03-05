import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

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
scaler = MinMaxScaler()
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
SEQ_LENGTH = 24
X_seq, y_seq = create_sequences(data_scaled, SEQ_LENGTH)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Definir el modelo LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, data_scaled.shape[1])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # Predicción de un solo valor (Usage_kWh)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluación en el conjunto de prueba
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")

# Hacer predicciones
predictions = model.predict(X_test)

# Invertir la normalización para obtener valores reales
y_test_real = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), X_exog.shape[1]-1)))))[:, 0]
predictions_real = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), X_exog.shape[1]-1)))))[:, 0]

# Calcular MSE y R2 usando los valores originales (desnormalizados) de y_test y predictions
mse = mean_squared_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)

if 0:
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    # Visualizar predicciones
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label="Real", color='blue')
    plt.plot(predictions_real, label="Predicho", color='red')
    plt.legend()
    plt.title("Predicción de Consumo Energético con LSTM")
    plt.show()

