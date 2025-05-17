from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout, Dense, Flatten

def seleccionModelosHibridos(flagModelo, num_features, SEQ_LENGTH, dropout):
    """Selecciona modelos híbridos LSTM-CNN según combinaciones predefinidas."""
    
    def modelo_1():
        # 1 LSTM + 1 CNN
        layers = [
            LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'), #características locales
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(1)
        ]
        return Sequential([layer for layer in layers if layer is not None])
    
    def modelo_2():
        # 2 LSTM + 1 CNN
        layers = [
            LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            LSTM(64, return_sequences=True),
            Dropout(0.2) if dropout else None,
            Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'), #características locales
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(1)
        ]
        return Sequential([layer for layer in layers if layer is not None])
    
    def modelo_3():
        # 1 LSTM + 2 CNN
        layers = [
            LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'), #características locales
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'), # patrones mas complejos
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(1)
        ]
        return Sequential([layer for layer in layers if layer is not None])
    
    def modelo_4():
        # 2 LSTM + 2 CNN
        layers = [
            LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            LSTM(64, return_sequences=True),
            Dropout(0.2) if dropout else None,
            Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'), #características locales
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'), # patrones mas complejos
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(1)
        ]
        return Sequential([layer for layer in layers if layer is not None])
    
    def modelo_5():
        # 3 LSTM + 3 CNN
        layers = [
            LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            LSTM(64, return_sequences=True),
            Dropout(0.2) if dropout else None,
            LSTM(32, return_sequences=True),
            Dropout(0.2) if dropout else None,
            Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'), #características locales
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'), # patrones mas complejos
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'), #características locales
            # MaxPooling1D(2), #Me da error ya que se resuce demasiado la dimension temporal 
            Flatten(),
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(1)
        ]
        return Sequential([layer for layer in layers if layer is not None])

    switch_modelos = {
        1: modelo_1,  # 1LSTM-1CNN
        2: modelo_2,  # 2LSTM-1CNN
        3: modelo_3,  # 1LSTM-2CNN
        4: modelo_4,  # 2LSTM-2CNN
        5: modelo_5   # 3LSTM-3CNN
    }

    return switch_modelos.get(flagModelo, modelo_1)()