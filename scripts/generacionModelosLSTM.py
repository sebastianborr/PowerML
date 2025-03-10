# Algoritmo para la creación de un modelo de LSTM, en función del numero de capas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def seleccionModelos(flagModelo, num_features, SEQ_LENGTH, dropout):
    """Selecciona y devuelve un modelo LSTM según el número de capas y si tiene Dropout."""

    def modelo_1():
        layers = [
            LSTM(128, return_sequences=False, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            Dense(8, activation='relu'),
            Dense(1)
        ]
        model = Sequential([layer for layer in layers if layer is not None])
        return model

    def modelo_2():
        layers = [
            LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            LSTM(64, return_sequences=False),
            Dropout(0.2) if dropout else None,
            Dense(8, activation='relu'),
            Dense(1)
        ]
        model = Sequential([layer for layer in layers if layer is not None])
        return model

    def modelo_3():
        layers = [
            LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            LSTM(64, return_sequences=True),
            Dropout(0.2) if dropout else None,
            LSTM(32, return_sequences=False),
            Dropout(0.2) if dropout else None,
            Dense(8, activation='relu'),
            Dense(1)
        ]
        model = Sequential([layer for layer in layers if layer is not None])
        return model

    def modelo_4():
        layers = [
            LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            LSTM(64, return_sequences=True),
            Dropout(0.2) if dropout else None,
            LSTM(32, return_sequences=True),
            Dropout(0.2) if dropout else None,
            LSTM(16, return_sequences=False),
            Dropout(0.2) if dropout else None,
            Dense(8, activation='relu'),
            Dense(1)
        ]
        model = Sequential([layer for layer in layers if layer is not None])
        return model

    def modelo_5():
        layers = [
            LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, num_features)),
            Dropout(0.2) if dropout else None,
            LSTM(96, return_sequences=True),
            Dropout(0.2) if dropout else None,
            LSTM(64, return_sequences=True),
            Dropout(0.2) if dropout else None,
            LSTM(32, return_sequences=True),
            Dropout(0.2) if dropout else None,
            LSTM(16, return_sequences=False),
            Dropout(0.2) if dropout else None,
            Dense(8, activation='relu'),
            Dense(1)
        ]
        model = Sequential([layer for layer in layers if layer is not None])
        return model

    # Switch manual con diccionario de funciones
    switch_modelos = {
        1: modelo_1,
        2: modelo_2,
        3: modelo_3,
        4: modelo_4,
        5: modelo_5
    }

    # Obtener modelo seleccionado y compilar
    model = switch_modelos.get(flagModelo, modelo_1)()  # Default: modelo_1 si flagModelo es inválido. Hace la llamada al modelo en cuestión
    #model.compile(optimizer='adam', loss='mse')

    return model

# Prueba del script
#model = seleccionModelos(flagModelo=5, num_features=10, SEQ_LENGTH=32, dropout=False)
#model.summary()