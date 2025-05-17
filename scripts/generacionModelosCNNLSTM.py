from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, Flatten

def seleccionModelosCNN_LSTM(flagModelo, num_features, SEQ_LENGTH, dropout):
    """Selecciona modelos híbridos CNN-LSTM según combinaciones predefinidas."""
    
    def modelo_1():
        # 1 CNN + 1 LSTM
        layers = [
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(SEQ_LENGTH, num_features)),
            MaxPooling1D(pool_size=2),
            Dropout(0.2) if dropout else None,
            LSTM(128, return_sequences=False),
            Dropout(0.2) if dropout else None,
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(1)
        ]
        return Sequential([layer for layer in layers if layer is not None])
    
    def modelo_2():
        # 2 CNN + 1 LSTM
        layers = [
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(SEQ_LENGTH, num_features)),
            MaxPooling1D(pool_size=2),
            Dropout(0.2) if dropout else None,
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2) if dropout else None,
            LSTM(128, return_sequences=False),
            Dropout(0.2) if dropout else None,
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(1)
        ]
        return Sequential([layer for layer in layers if layer is not None])
    
    def modelo_3():
        # 1 CNN + 2 LSTM
        layers = [
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(SEQ_LENGTH, num_features)),
            MaxPooling1D(pool_size=2),
            Dropout(0.2) if dropout else None,
            LSTM(128, return_sequences=True),
            Dropout(0.2) if dropout else None,
            LSTM(64, return_sequences=False),
            Dropout(0.2) if dropout else None,
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(1)
        ]
        return Sequential([layer for layer in layers if layer is not None])
    
    def modelo_4():
        # 2 CNN + 2 LSTM
        layers = [
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(SEQ_LENGTH, num_features)),
            MaxPooling1D(pool_size=2),
            Dropout(0.2) if dropout else None,
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2) if dropout else None,
            LSTM(128, return_sequences=True),
            Dropout(0.2) if dropout else None,
            LSTM(64, return_sequences=False),
            Dropout(0.2) if dropout else None,
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(1)
        ]
        return Sequential([layer for layer in layers if layer is not None])
    
    switch_modelos = {
        1: modelo_1,  # 1CNN-1LSTM
        2: modelo_2,  # 2CNN-1LSTM
        3: modelo_3,  # 1CNN-2LSTM
        4: modelo_4   # 2CNN-2LSTM
    }

    return switch_modelos.get(flagModelo, modelo_1)()