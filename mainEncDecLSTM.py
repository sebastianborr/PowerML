import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Conv1D, MaxPooling1D, Dropout, Dense, Input, 
                                     Bidirectional, Multiply, Concatenate, Layer, TimeDistributed, RepeatVector)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from time import time
# Importar la función de preprocesamiento de datos
from scripts.preprocess_data_Tetuan import preprocess_data_electricity

#------------------------------------------
# Cargar datos
df = pd.read_csv('data/02_Tetuan_City_power_consumption.csv')

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

# Parámetros
numHoras = 8
SEQ_LENGTH = 6* numHoras  # cada 10 mins los datos
X_seq, y_seq = create_sequences(X_exog, SEQ_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=41)
PRED_HORIZON = 1
NUM_FEATURES = X_exog.shape[1]-2  # Número de características de entrada

dropout_rate = 0.2  # Tasa de dropout
epoch_rate = 100  # Número de épocas
batch_size_rate = 64  # Tamaño del lote

# Definir el modelo encoder-decoder con atención (paper 39)
# Capa de atención personalizada para encoder-decoder
class Seq2SeqAttention(Layer):
    def __init__(self, units):
        super(Seq2SeqAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, encoder_output, decoder_hidden):
        # Score = d_{t-1}^T * h_i (Eq. 8)
        decoder_hidden = tf.expand_dims(decoder_hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(encoder_output) + self.W2(decoder_hidden)))
        attention_weights = tf.nn.softmax(score, axis=1)  # Eq. 9
        context_vector = attention_weights * encoder_output  # Eq. 10
        return tf.reduce_sum(context_vector, axis=1)

# Arquitectura encoder-decoder con atención
def build_encoder_decoder_attention_model(seq_length, num_features, pred_horizon):
    # Encoder
    encoder_inputs = Input(shape=(seq_length, num_features))
    encoder_lstm = LSTM(128, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    
    # Decoder con atención
    decoder_inputs = RepeatVector(pred_horizon)(state_h)
    decoder_lstm = LSTM(128, return_sequences=True, return_state=False)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    
    # Mecanismo de atención (Eqs. 7-10)
    attention = Seq2SeqAttention(64)
    context_vector = attention(encoder_outputs, decoder_outputs)
    
    # Salida para intervalo [LB, UB] (Granulación difusa se aplica externamente)
    output = Dense(1)(context_vector)  # 2 salidas: Límite inferior y superior
    
    return Model(encoder_inputs, output)

# Uso
model = build_encoder_decoder_attention_model(SEQ_LENGTH, NUM_FEATURES, PRED_HORIZON)
# Definir el optimizador Adam con un learning rate específico
optimizer = Adam(learning_rate=0.001)
# Compilar el modelo
model.compile(optimizer=optimizer, loss='mse')
# Resumen del modelo
model.summary()

start_time = time()  # Registrar el tiempo de inicio
# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=epoch_rate, batch_size=batch_size_rate, validation_data=(X_test, y_test))
end_time = time()  # Registrar el tiempo de fin
training_time = end_time - start_time  # Calcular el tiempo total de entrenamiento

# Hacer predicciones
predictions = model.predict(X_test)

# Invertir la normalización para obtener valores reales
y_test_real = scaler_y.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 2)))))[:, 0]
# predictions_real = scaler_y.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), 2)))))[:, 0]

# Extraer el último timestep de la salida (forma: (n, 1))
pred_last = predictions[:, -1, :]
#pred_last = predictions

# Invertir la normalización para obtener valores reales
predictions_real = scaler_y.inverse_transform(np.hstack((pred_last, np.zeros((len(pred_last), 2)))))[:, 0]

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
    if 1:
        results = []
        results.append({
            'num_layers': 'EncDec_LSTM',
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
        results_df.to_csv('./output/ResultsDL/01_EncDecLSTM_Tetuan_Paper.csv', index=False)